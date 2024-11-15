# copyright:https://github.com/microsoft/Semi-supervised-learning
# This code is modified based on Freematch code
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConsistencyLoss:

    def __call__(self, logits, targets, mask=None, weight=None):
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none', weight=weight)
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()


class SelfAdaptiveThresholdLoss:

    def __init__(self, sat_ema, ):
        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()

    @torch.no_grad()
    def __update__params__(self, logits_ulb_w, tau_t, p_t, label_hist):
        # Updating the histogram for the SAF losses here so that I dont have to call the torch.no_grad() function again.
        # You can do it in the SAF losses also, but without accumulating the gradient through the weak augmented logits

        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)

        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()

        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        hist_norm = histogram / histogram.sum()
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * hist_norm

        last = torch.abs((probs_ulb_w.mean(dim=0) - hist_norm) / hist_norm)
        last[last == float('inf')] = 1
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * last

        return tau_t, p_t, label_hist, histogram

    def __call__(self, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist, mask_ratio, mask_ratio_all, ):
        tau_t, p_t, label_hist, histogram = self.__update__params__(logits_ulb_w, tau_t, p_t, label_hist)

        logits_ulb_w = logits_ulb_w.detach()
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)

        tau_t_c = 1 - torch.abs(p_t)
        mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).to(max_probs_w.dtype)

        selected_labels = max_idx_w[mask == 1]
        hist_mask = torch.bincount(selected_labels, minlength=logits_ulb_w.shape[1]).to(logits_ulb_w.dtype)
        mask_label = hist_mask / hist_mask.sum()
        mask_ratio_last = hist_mask / histogram
        mask_ratio_last[torch.isnan(mask_ratio_last)] = 0.0
        mask_ratio = mask_ratio * self.sat_ema + (1. - self.sat_ema) * mask_ratio_last
        trust = mask.clone()
        partial_trust = ((max_probs_w > probs_ulb_w.mean(dim=0)[max_idx_w]) & (mask == 0)).to(max_probs_w.dtype)
        partial_idx = max_idx_w[partial_trust == 1]
        partial_mask = torch.bincount(partial_idx, minlength=logits_ulb_w.shape[1]).to(logits_ulb_w.dtype)
        no_trust = ((partial_trust == 0) & (mask == 0)).to(max_probs_w.dtype)
        no_idx = max_idx_w[no_trust == 1]
        no_mask = torch.bincount(no_idx, minlength=logits_ulb_w.shape[1]).to(logits_ulb_w.dtype)
        partial_mask_all = partial_mask / logits_ulb_w.shape[0]
        no_mask_all = no_mask / logits_ulb_w.shape[0]
        mask_ratio_all = mask_ratio_all * self.sat_ema + (1. - self.sat_ema) * hist_mask / logits_ulb_w.shape[0]
        mask_unb = mask.clone()

        if not torch.isnan(mask_label).any().item():
            weight = 1 / mask_ratio_all
            weight[weight == float('inf')] = 0.0
            weight = (weight / torch.max(weight, dim=-1)[0])
            for i in range(p_t.shape[0]):
                mask_modify = (max_idx_w == i) & (mask == 1)  # 找到需要修改的位置
                num_to_modify = int(torch.sum(mask_modify) * (1 - weight[i]))  # 计算需要修改的数量
                if num_to_modify > 1:
                    indices_to_modify = torch.nonzero(mask_modify, as_tuple=False).squeeze()  # 找到需要修改的索引
                    indices_to_modify = indices_to_modify[torch.randperm(indices_to_modify.size(0))][
                                        :num_to_modify - 1]  # 随机选择需要修改的索引
                    mask[indices_to_modify] = False  # 修改b张量

        loss = self.criterion(logits_ulb_s, max_idx_w, mask=mask)
        return loss, mask, tau_t, p_t, label_hist, mask_label, mask_ratio, mask_ratio_all, mask_unb, max_idx_w, partial_trust, partial_mask_all, no_trust, trust, no_mask_all
