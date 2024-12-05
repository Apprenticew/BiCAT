import os
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC, ROC, AveragePrecision, PrecisionRecallCurve
from torchmetrics.classification import MulticlassAccuracy
from indicator import evaluating_indicator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Record:
    def __init__(self, args, class_num, class_counts):
        self.best_value = torch.Tensor([0]).to(device)
        self.best_iter = 0
        self.best_value2 = torch.Tensor([0]).to(device)
        self.best_iter2 = 0
        self.best_value3 = torch.Tensor([0]).to(device)
        self.best_iter3 = 0
        self.best_value4 = torch.Tensor([0]).to(device)
        self.best_iter4 = 0
        self.f1 = 0
        self.args = args
        self.class_num = class_num
        if args.test:
            self.model_path = os.path.join(args.model_path, args.ssl_weights_path.replace('/', '>'),
                                           os.path.basename(args.folder_arbitrary_path), )
        else:
            self.model_path = args.model_path

        self.class_counts = class_counts
        self.roc_auc = torch.Tensor()
        self.micro_roc_auc, self.roc, self.roc_curve, self.prc, self.prc_curve = None, None, None, None, None
        self.prc_auc = torch.Tensor()
        self.y_true, self.y_score, self.y_pred, self.score_norma, self.score_norma2 = [], [], [], [], []
        self.eval_dict, self.k_acc = {}, {}
        self.metric_model = None
        self.eval_ind = None
        self.confusion_matrix = None
        self.set_default()

    def set_default(self):
        self.eval_ind = evaluating_indicator(self.class_num)
        self.y_true, self.y_score, self.y_pred, self.score_norma, self.score_norma2 = [], [], [], [], []
        if self.class_num != 2:
            self.roc = AUROC(task="multiclass", num_classes=self.class_num, average=None).to(device)
            self.roc_curve = ROC(task="multiclass", num_classes=self.class_num, average='micro').to(device)
            # self.roc_curve = ROC(task="multiclass", num_classes=self.class_num, average=None).to(device)
            self.prc = AveragePrecision(task="multiclass", num_classes=self.class_num, average=None).to(device)
            self.prc_curve = PrecisionRecallCurve(task="multiclass", num_classes=self.class_num).to(device)
            self.metric_model = MulticlassAccuracy(average="micro", num_classes=self.class_num).to(device)
        else:
            self.roc = AUROC(task="binary").to(device)
            self.roc_curve = ROC(task="binary").to(device)
            self.prc = AveragePrecision(task="binary").to(device)
            self.prc_curve = PrecisionRecallCurve(task="binary").to(device)
            self.metric_model = MulticlassAccuracy(average="micro", num_classes=self.class_num).to(device)

    def update(self, score_norma, label, ssl=False, tau_t=None, p_t=None, mode=None):
        prob, label_pred = torch.max(score_norma, dim=1)

        if mode == 'conservative':
            threshold = 0.9
            max_prob, max_idx = score_norma[:, :self.class_num - 1].max(dim=1)
            label_pred = torch.where(score_norma[:, self.class_num - 1] < threshold, max_idx,
                                     torch.tensor(self.class_num - 1))

        self.y_pred += list(label_pred)
        self.y_true += list(label)
        self.y_score += list(prob)
        self.score_norma.append(score_norma)
        if self.class_num == 2:
            score_norma = score_norma[:, 1]
            #     score_norma = score_norma[:, 1:-1]
        self.score_norma2.append(score_norma)
        # birads all
        if ssl:
            # x = torch.tensor([0.909, 0.873, 0.875, 0.839]).to(device)
            # p_t= torch.tensor([0.909, 0.873, 0.875, 0.839]).to(device)
            y = (1 - p_t)
            mask = (prob >= tau_t * y[label_pred]) * (prob > torch.tensor([0.5]).to(device))
            # mask = prob >= 0.6
            prob = prob[mask]
            score_norma = score_norma[mask]
            label_pred = label_pred[mask]
            label = label[mask]
        if not label_pred.numel() == 0:
            self.metric_model.update(label_pred, label)
            self.eval_ind.update_confusion_matrix(label_pred, label)
            self.roc.update(score_norma, label)
            self.prc.update(score_norma, label)
            self.roc_curve.update(score_norma, label)
            self.prc_curve.update(score_norma, label)
        else:
            print('No data in the batch meets the requirements.')

        # if self.class_num != 2:
        #     self.y_score.append(score_norma)
        # else:
        #     self.y_score.append(score_norma[:, 1])

    def write(self, curr_iter, bs=False):
        os.makedirs(self.model_path, exist_ok=True)
        # self.eval_ind.summerize_accuracy()
        # self.eval_dict, self.k_acc = self.eval_ind.summerize_confusion_matrix(), self.eval_ind.k_acc[1]
        self.eval_dict = self.eval_ind.summerize_confusion_matrix(f1_type='weighted', weights=self.class_counts)
        self.confusion_matrix = self.eval_ind.confusion_matrix
        self.k_acc = self.metric_model.compute()
        # print(self.k_acc)
        self.roc_auc = self.roc.compute()
        self.prc_auc = self.prc.compute()
        # self.roc_curve.plot(score=True)
        # precision, recall, _ = self.prc_curve.compute()
        if self.class_num != 2:
            self.f1 = self.eval_dict["F1_summary"]
            self.micro_roc_auc = roc_auc_score(torch.tensor(self.y_true).cpu().numpy(),
                                               torch.cat(self.score_norma, dim=0).cpu().numpy(),
                                               average='micro',
                                               multi_class='ovr')
        else:
            # self.f1 = self.eval_dict['F1'][1]
            self.f1 = self.eval_dict["F1_summary"]
            self.micro_roc_auc = roc_auc_score(torch.tensor(self.y_true).cpu().numpy(),
                                               torch.cat(self.score_norma2, dim=0).cpu().numpy(),
                                               )
        if self.best_value < self.f1:
            self.best_value = self.f1
            self.best_iter = curr_iter
        # if self.best_value2 < self.roc_auc.mean():
        #     self.best_value2 = self.roc_auc.mean()
        #     self.best_iter2 = curr_iter
        if self.best_value2 < self.micro_roc_auc:
            self.best_value2 = self.micro_roc_auc
            self.best_iter2 = curr_iter
        if self.best_value3 < self.prc_auc.mean():
            self.best_value3 = self.prc_auc.mean()
            self.best_iter3 = curr_iter
        if self.best_value4 < self.k_acc:
            self.best_value4 = self.k_acc
            self.best_iter4 = curr_iter
        # print('\n')
        # write = True
        if not bs:
            file_path = os.path.join(self.model_path, '++pred_classify.txt')
            with open(file_path, 'a', encoding='utf-8') as f:
                model_type = "Model"
                # torch.set_printoptions(precision=3)
                if curr_iter != 0:
                    f.write(f"{model_type} validating,iter:{curr_iter}\n")
                else:
                    f.write('\nTest dataset. Path: %s\n' % self.args.folder_arbitrary_path)
                    f.write('Class number: %d\n' % self.class_num)
                    if self.args.load_ssl_weights:
                        f.write(f'{model_type} loaded from checkpoint. Path: %s\n' % self.args.ssl_weights_path)

                for key, value in self.eval_dict.items():
                    # value = value.item()
                    # value = torch.round(value, decimals=3)
                    f.write(f"{key}: {value}\n")
                f.write(f"roc_auc_ave: {self.roc_auc.mean():.3f}\n")
                f.write(f"micro_roc_auc: {self.micro_roc_auc:.3f}\n")
                f.write(f"roc_auc: {self.roc_auc}\n")
                f.write(f"prc_auc_ave: {self.prc_auc.mean():.3f}\n")
                f.write(f"prc_auc: {self.prc_auc}\n")
                f.write(f"k_acc: {self.k_acc}\n")

                if curr_iter != 0:
                    f.write(f"max_F1: {self.best_value.item():.3f},iter/step:{self.best_iter}/{curr_iter}\n")
                    f.write(f"max_roc_auc: {self.best_value2:.3f},iter/step:{self.best_iter2}/{curr_iter}\n")
                    f.write(f"max_prc_auc: {self.best_value3.item():.3f},iter/step:{self.best_iter3}/{curr_iter}\n")
                    f.write(f"max_k_acc: {self.best_value4.item():.3f},iter/step:{self.best_iter4}/{curr_iter}\n\n")
                else:
                    print()
                    self.plot_confusion()

    def auc(self):
        y_score, y_true = torch.stack(self.y_score, 0), torch.stack(self.y_true, 0)
        y_score = y_score.view(-1, y_score.size(-1))
        y_true = y_true.view(-1)

    def plot_confusion(self, ):
        recall_num = torch.sum(self.confusion_matrix, dim=1)
        precision_num = torch.sum(self.confusion_matrix, dim=0)
        sns.set()

        recall_path = os.path.join(self.model_path, '+Recall.png')
        precision_path = os.path.join(self.model_path, '+Precision.png')
        prc_path = os.path.join(self.model_path, '+PRC.png')
        roc_path = os.path.join(self.model_path, '+ROC.png')
        roc_excel_path = os.path.join(self.model_path, '+ROC.xlsx')
        prc_excel_path = os.path.join(self.model_path, '+PRC.xlsx')
        confusion_matrix_path = os.path.join(self.model_path, '+ConfusionMatrix.png')

        plt.figure()
        recall_ax = sns.heatmap(self.confusion_matrix / recall_num.unsqueeze(1), annot=True)
        recall_ax.set_title('Recall Confusion Matrix')
        recall_ax.set_xlabel('Predicted Label')
        recall_ax.set_ylabel('True Label')
        recall_ax.set_xticklabels(range(1, self.confusion_matrix.shape[1] + 1))  # 设置横坐标刻度为整数
        recall_ax.set_yticklabels(range(1, self.confusion_matrix.shape[0] + 1))  # 设置纵坐标刻度为整数
        plt.savefig(recall_path)
        plt.show()

        plt.figure()
        precision_ax = sns.heatmap(self.confusion_matrix / precision_num.unsqueeze(0), annot=True)
        precision_ax.set_title('Precision Confusion Matrix')
        precision_ax.set_xlabel('Predicted Label')
        precision_ax.set_ylabel('True Label')
        precision_ax.set_xticklabels(range(1, self.confusion_matrix.shape[1] + 1))  # 设置横坐标刻度为整数
        precision_ax.set_yticklabels(range(1, self.confusion_matrix.shape[0] + 1))  # 设置纵坐标刻度为整数
        plt.savefig(precision_path)
        plt.show()

        plt.figure()
        precision_ax = sns.heatmap(self.confusion_matrix, annot=True, fmt='.0f')
        precision_ax.set_title('Confusion Matrix')
        precision_ax.set_xlabel('Predicted Label')
        precision_ax.set_ylabel('True Label')
        precision_ax.set_xticklabels(range(1, self.confusion_matrix.shape[1] + 1))  # 设置横坐标刻度为整数
        precision_ax.set_yticklabels(range(1, self.confusion_matrix.shape[0] + 1))  # 设置纵坐标刻度为整数
        plt.savefig(confusion_matrix_path)
        plt.show()

        # precision = precision[0]
        # recall = recall[0]
        precision, recall, _ = self.prc_curve.compute()
        plt.figure()
        if self.class_num == 2:
            plt.step(recall.cpu(), precision.cpu(), where='post', label=f'Class')
        else:
            for class_idx in range(self.class_num):
                plt.step(recall[class_idx].cpu(), precision[class_idx].cpu(), where='post',
                         label=f'Class {class_idx + 1}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(prc_path)
        plt.show()

        # precision = precision[0]
        # recall = recall[0]
        fpr, sensitivity, _ = self.roc_curve.compute()
        plt.figure()
        if self.class_num == 2:
            plt.step(fpr.cpu(), sensitivity.cpu(), where='post', label=f'Class ')
        else:
            # for class_idx in range(self.class_num):
            #     plt.step(fpr[class_idx].cpu(), sensitivity[class_idx].cpu(), where='post',
            #              label=f'Class {class_idx + 1}')
            plt.step(fpr.cpu(), sensitivity.cpu(), where='post', label=f'Class ')
        plt.xlabel('1-specificity')
        plt.ylabel('sensitivity')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(roc_path)
        plt.show()

        # fpr = torch.stack(fpr).cpu().numpy()
        # sensitivity = torch.stack(sensitivity).cpu().numpy()

        # roc curve
        if self.class_num == 2:
            fpr = fpr.cpu()
            sensitivity = sensitivity.cpu()
        else:
            # fpr = fpr[-1].cpu().numpy()
            # sensitivity = sensitivity[-1].cpu().numpy()
            fpr = fpr.cpu()
            sensitivity = sensitivity.cpu()
        # 假阳性率（False Positive Rate）和灵敏度（Sensitivity）数据
        data = {'1-specificity': fpr, 'Sensitivity': sensitivity}
        # data = {'1-specificity': fpr.cpu(), 'Sensitivity': sensitivity.cpu()}
        # 创建DataFrame
        # df = pd.DataFrame(data.T, columns=['Column1', 'Column2', 'Column3', 'Column4'])
        df = pd.DataFrame(data)
        # 将DataFrame保存为Excel文件
        df.to_excel(roc_excel_path, index=False)

        # # prc curve
        # precision = precision[3].cpu().numpy()
        # recall = recall[3].cpu().numpy()
        # # 假阳性率（False Positive Rate）和灵敏度（Sensitivity）数据
        # data = {'recall': recall, 'precision': precision}
        # # 创建DataFrame
        # # df = pd.DataFrame(data.T, columns=['Column1', 'Column2', 'Column3', 'Column4'])
        # df = pd.DataFrame(data)
        # # 将DataFrame保存为Excel文件
        # df.to_excel(prc_excel_path, index=False)

        # # 创建包含 fpr 数据的 DataFrame
        # df_fpr = pd.DataFrame(fpr.T, columns=['FPR_Class1', 'FPR_Class2', 'FPR_Class3', 'FPR_Class4'])
        # # 创建包含 sensitivity 数据的 DataFrame
        # df_sensitivity = pd.DataFrame(sensitivity.T,
        #                               columns=['Sensitivity_Class1', 'Sensitivity_Class2', 'Sensitivity_Class3',
        #                                        'Sensitivity_Class4'])
        #
        # df_fpr.to_excel(roc_excel_path, sheet_name='FPR', index=False)
        # df_sensitivity.to_excel(roc_excel_path, sheet_name='Sensitivity', index=False)

        # for class_idx in range(self.args.class_num):
        #     # plt.step(fpr[class_idx].cpu(), sensitivity[class_idx].cpu(), where='post', label=f'Class {class_idx + 1}')
        #     df_fpr = pd.DataFrame(fpr[class_idx].cpu(), columns=[f'FPR_Class {class_idx + 1}'])
        #     df_sensitivity = pd.DataFrame(sensitivity[class_idx].cpu(), columns=[f'Sensitivity_Class {class_idx + 1}'])
        #     df_fpr.to_excel(roc_excel_path, sheet_name=f'Class {class_idx + 1}', index=False)
        #     df_sensitivity.to_excel(roc_excel_path, sheet_name=f'Class {class_idx + 1}', index=False)
        #
        # # 创建包含所有数据的 DataFrame
        # df = pd.DataFrame({
        #     'FPR_Class1': fpr[0].cpu().numpy(),
        #     'FPR_Class2': fpr[1].cpu().numpy(),
        #     'FPR_Class3': fpr[2].cpu().numpy(),
        #     'FPR_Class4': fpr[3].cpu().numpy(),
        #     'Sensitivity_Class1': sensitivity[0].cpu().numpy(),
        #     'Sensitivity_Class2': sensitivity[1].cpu().numpy(),
        #     'Sensitivity_Class3': sensitivity[2].cpu().numpy(),
        #     'Sensitivity_Class4': sensitivity[3].cpu().numpy(),
        # })
        #
        # # 创建 ExcelWriter 对象
        # with pd.ExcelWriter(roc_excel_path, engine='openpyxl') as writer:
        #     # 将数据写入主表
        #     df.to_excel(writer, index=False, sheet_name='Main')
        #
        #     # 创建四个子表
        #     for i in range(4):
        #         sub_df = df[['FPR_Class{}'.format(i + 1), 'Sensitivity_Class{}'.format(i + 1)]]
        #         sub_df.to_excel(writer, index=False, sheet_name='Class{}'.format(i + 1))
