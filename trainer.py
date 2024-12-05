import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from ema import EMA
from tqdm import tqdm
from contextlib import nullcontext
from losses import SelfAdaptiveThresholdLoss

from record import Record

AMP_ENABLED = True
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer(model, train_dataloader, unlabel_dataloader, test_dataloader, class_counts_all, args):
    # 日志信息保存路径
    logger = SummaryWriter(os.path.join(os.path.dirname(args.model_path), 'logger', os.path.basename(args.model_path)))
    class_counts = class_counts_all[0]
    print(class_counts_all[:])
    record = Record(args, class_num=args.class_num, class_counts=class_counts_all[1])
    amp = nullcontext
    if AMP_ENABLED:
        scaler = GradScaler()
        amp = autocast

    # 训练网络
    ema_val = 0.999
    set_ema = 0.999
    curr_iter = 1
    num_eval_iters = int(sum(class_counts) / args.bsize)
    ulb_loss_ratio = 1.0
    lb_loss_ratio = 1.0

    p_t = torch.zeros(args.class_num)
    label_hist = torch.ones(args.class_num) / args.class_num
    tau_t = torch.ones(args.class_num).mean()

    mask_ratio = torch.zeros(args.class_num)
    mask_ratio_all = torch.ones(args.class_num) / args.class_num

    sat_criterion = SelfAdaptiveThresholdLoss(set_ema, )
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer1 = optim.NAdam([{'params': model.parameters()}, ], lr=args.lr)

    sched = lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_eval_iters * 30, eta_min=1e-4)

    if args.load_ssl_weights:
        ckpt = torch.load(args.ssl_weights_path)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        # Algorithm specfic loading
        curr_iter = ckpt['curr_iter'] + 1
        tau_t = ckpt['tau_t']
        p_t = ckpt['p_t']
        mask_ratio = ckpt['mask_ratio']
        mask_ratio_all = ckpt['mask_ratio_all']
        label_hist = ckpt['label_hist']

        print('Initialized checkpoint parameters..')
        print('Model loaded from checkpoint. Path: %s' % args.ssl_weights_path)

    p_t = p_t.to(device)
    label_hist = label_hist.to(device)
    tau_t = tau_t.to(device)
    mask_ratio = mask_ratio.to(device)
    mask_ratio_all = mask_ratio_all.to(device)

    model.train()
    net = EMA(model=model, decay=ema_val)
    net.train()
    ssl = False
    ssl2 = False

    loop = tqdm(zip(train_dataloader, unlabel_dataloader), desc=f'iter [{curr_iter}/{args.num_iters}]')
    for (batch_lb, batch_ulb) in loop:
        img_lb_w, label = batch_lb[0].to(device), batch_lb[1].to(device)
        img_ulb_w, img_ulb_s = batch_ulb[0].to(device), batch_ulb[1].to(device)
        img = torch.cat([img_lb_w, img_ulb_w, img_ulb_s])

        num_lb = img_lb_w.shape[0]
        num_ulb = img_ulb_w.shape[0]

        assert num_ulb == img_ulb_s.shape[0]

        if curr_iter >= num_eval_iters * 10:
            ssl = True
        with amp('cuda'):

            score = net(img)
            logits_lb = score[:num_lb]
            logits_ulb_w, logits_ulb_s = score[num_lb:].chunk(2)

            loss_lb = criterion(logits_lb, label)
            if ssl:
                loss_sat, mask, tau_t, p_t, label_hist, mask_label, mask_ratio, mask_ratio_all, _, max_idx_w, partial_trust, partial_mask_all, no_trust, trust, no_mask_all = sat_criterion(
                    logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist, mask_ratio, mask_ratio_all
                )

            if ssl:
                loss = lb_loss_ratio * loss_lb + ulb_loss_ratio * loss_sat
            else:
                loss = lb_loss_ratio * loss_lb

        if AMP_ENABLED:
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.update()
        else:
            loss.backward()
            optimizer1.step()
        if ssl:
            sched.step()

        net.update()
        model.zero_grad()

        loop.set_description(f'iter [{curr_iter}/{args.num_iters}]')
        print(f'Loss: {loss.item()},Label:{label.long() + 1}')
        # Logging in tensorboard
        logger.add_scalar('train/lr', optimizer1.param_groups[0]['lr'], curr_iter)
        logger.add_scalar('train/lb_loss', loss_lb.item(), curr_iter)
        logger.add_scalar('train/total_loss', loss.item(), curr_iter)
        logger.add_scalar('train/lb_loss_ratio', lb_loss_ratio, curr_iter)
        logger.add_scalar('train/tau_t', tau_t.item(), curr_iter)

        if ssl:
            logger.add_scalar('train/sat_loss', loss_sat.item(), curr_iter)
            logger.add_scalar('mask/mask', 1 - mask.mean().item(), curr_iter)
            logger.add_scalars(main_tag='mask/mask',
                               tag_scalar_dict={
                                   f'mask{i + 1}': mask_label[i] for i in range(args.class_num)
                               },
                               global_step=curr_iter)
            logger.add_scalars(main_tag='mask/trust_mask',
                               tag_scalar_dict={
                                   f'trust_mask': trust.mean().item(),
                                   f'partial_mask': partial_trust.mean().item(),
                                   f'no_mask': no_trust.mean().item(),
                               },
                               global_step=curr_iter)
            logger.add_scalars(main_tag='mask/partial_mask_all',
                               tag_scalar_dict={
                                   f'partial_mask_all{i + 1}': partial_mask_all[i] for i in range(args.class_num)
                               },
                               global_step=curr_iter)
            logger.add_scalars(main_tag='mask/no_mask_all',
                               tag_scalar_dict={
                                   f'no_mask_all{i + 1}': no_mask_all[i] for i in range(args.class_num)
                               },
                               global_step=curr_iter)

        logger.add_scalar('train/p_t', p_t.mean().item(), curr_iter)
        logger.add_scalars(main_tag='train/p_t',
                           tag_scalar_dict={
                               f'p_t{i + 1}': p_t[i] for i in range(args.class_num)
                           },
                           global_step=curr_iter)
        logger.add_scalars(main_tag='mask/label_hist',
                           tag_scalar_dict={
                               f'label_hist{i + 1}': label_hist[i] for i in range(args.class_num)
                           },
                           global_step=curr_iter)
        logger.add_scalars(main_tag='mask/mask_ratio',
                           tag_scalar_dict={
                               f'mask_ratio{i + 1}': mask_ratio[i] for i in range(args.class_num)
                           },
                           global_step=curr_iter)
        logger.add_scalars(main_tag='mask/mask_ratio_all',
                           tag_scalar_dict={
                               f'mask_ratio_all{i + 1}': mask_ratio_all[i] for i in range(args.class_num)
                           },
                           global_step=curr_iter)

        # curr_iter += 1
        if (curr_iter) % num_eval_iters == 0:
            save_dict = {
                'model_state_dict': model.state_dict(),
                # 'ema_state_dict': net.state_dict(),
                # 'optimizer_state_dict': optimizer1.state_dict(),
                # 'curr_iter': curr_iter,
                'tau_t': tau_t.cpu(),
                'p_t': p_t.cpu(),
                # 'mask_ratio': mask_ratio.cpu(),
                # 'mask_ratio_all': mask_ratio_all.cpu(),
                # 'label_hist': label_hist.cpu(),
            }
            torch.save(save_dict, os.path.join(args.model_path, 'curr_iter_%d' % curr_iter))
            print('parameter is loaded successfully,start validation...\n')
            # net.eval()
            model.eval()
            # if curr_iter >= num_eval_iters * 20:
            #     ssl2 = True
            with torch.no_grad():
                for i, y in enumerate(test_dataloader):
                    inputs = y[0].to(device)
                    label = y[1].to(device)
                    score = model(inputs)
                    score_norma = nn.Softmax(dim=1)(score)
                    record.update(score_norma, label, ssl2, tau_t, p_t)

                record.write(curr_iter)
                record.set_default()

                logger.add_scalar('evaluate/roc_auc', record.micro_roc_auc, curr_iter)
                logger.add_scalar('evaluate/prc_auc', record.prc_auc.mean(), curr_iter)
                logger.add_scalar('evaluate/F1_summary', record.f1, curr_iter)
                logger.add_scalar('evaluate/k_acc', record.k_acc, curr_iter)
                logger.add_scalars(main_tag='evaluate/roc_auc_class',
                                   tag_scalar_dict={
                                       f'roc_auc_class{i + 1}': record.roc_auc[i] for i in range(args.class_num)
                                   },
                                   global_step=curr_iter)

            # net.train()
            model.train()
        curr_iter += 1
