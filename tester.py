import os
import gc
import pathlib
import seaborn as sns
from sklearn.metrics import roc_auc_score
from torch.multiprocessing import Pool, set_start_method
import shap
import time
from scipy import stats
from PIL import Image
from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as utils
from torchmetrics import AUROC
from torchmetrics.classification import MulticlassAccuracy
from ema import EMA
from indicator import evaluating_indicator
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from record import Record
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 128
mean, std = 0.345, 0.145
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((size, size), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def tester(model, arbitraty_dataloader, args, class_counts=None, class_num=4, bs=False, heatmap=True, mode='liberal'):
    print('parameter is loaded successfully,start validation...')
    record = Record(args, class_num=class_num, class_counts=class_counts)
    ssl = False

    ckpt = torch.load(args.ssl_weights_path, weights_only=True)
    tau_t = ckpt['tau_t'].to(device)
    p_t = ckpt['p_t'].to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    model_path = os.path.join(args.model_path, os.path.basename(args.folder_arbitrary_path), )
    os.makedirs(model_path, exist_ok=True)

    print(f'\n{args.folder_arbitrary_path}:{tau_t * (1 - p_t)}\n')

    target_layers = [model.resnet.conv5_x[-1]]
    start_time = time.time()

    if not bs:
        name_cluster = []
        for i, y in enumerate(arbitraty_dataloader):
            inputs = y[0].to(device)
            label = y[1].to(device)
            paths = y[2]

            try:
                if not heatmap:
                    with torch.no_grad():
                        score = model(inputs)
                else:
                    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                        grayscale_cams = cam(input_tensor=inputs)
                        score = cam.outputs.detach()  # Get the model outputs
                if class_num == 3:
                    score = score[:, 1:]
                elif class_num == 2:
                    score = score[:, 2:]
                score_norma = nn.Softmax(dim=1)(score)
                probabilities, preds = torch.max(score_norma, dim=1)  # Get the max probability and index

                record.update(score_norma, label, ssl=ssl, tau_t=tau_t, p_t=p_t, mode=mode)

                for j, image_path in enumerate(paths):
                    file_name = os.path.basename(image_path)
                    name = file_name.split(".")[0]
                    file_name = os.path.basename(os.path.dirname(image_path)) + '_' + file_name
                    name_cluster.append(name)

                    if heatmap:
                        label1 = label[j].squeeze().detach().cpu().numpy()
                        pred = preds[j].item()
                        probability = probabilities[j].item()
                        # Save the image
                        if pred == label1:
                            preddir = os.path.join(model_path, 'pred', str(pred + 1), 'True')
                        else:
                            preddir = os.path.join(model_path, 'pred', str(pred + 1), 'False')
                        os.makedirs(preddir, exist_ok=True)
                        image = Image.open(image_path)
                        image_crop = image
                        imgpath = os.path.join(preddir, file_name)
                        image_crop.save(imgpath)

                        # Save heatmap visualization
                        grayscale_cam = grayscale_cams[j]
                        grayscale_cam = ndimage.zoom(grayscale_cam, image_crop.size[0] / 128, order=1)
                        rgb_img = np.float32(image_crop) / 255

                        # 保存灰度热图
                        # grayscalepath = os.path.join(preddir, f'rgb_{file_name}')
                        # Image.fromarray((grayscale_cam * 255).astype(np.uint8)).save(grayscalepath)

                        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.8)
                        visual = Image.fromarray(visualization)
                        if probability > (tau_t * (1 - p_t))[pred]:
                            confidence_level = 'Highly_confident'
                        elif probability > 1 - 2 * (tau_t * (1 - p_t))[pred] and probability > (tau_t * (1 - p_t))[
                            pred]:
                            confidence_level = 'Confident'
                        else:
                            confidence_level = 'Not_confident'
                        visualpath = os.path.join(preddir, f'{confidence_level}_{file_name}')
                        visual.save(visualpath)
                        visual.close()

                        # Clean up
                        image.close()
                        image_crop.close()

            except RuntimeError as e:
                print(f"RuntimeError: {e}")

            finally:
                clear_all_gpu_memory()

        # record.write(0, bs=bs)

        pred_cluster = record.y_pred
        pred_cluster = [x + 1 for x in pred_cluster]
        prob_cluster = record.score_norma
        label_cluster = record.y_true
        label_cluster = [x + 1 for x in label_cluster]

        record.set_default()

        pred_cluster = torch.Tensor(pred_cluster).detach().cpu().numpy()
        prob_cluster = torch.cat(prob_cluster, dim=0).squeeze().cpu().numpy()
        label_cluster = torch.Tensor(label_cluster).detach().cpu().numpy()
        print(f"model Cohen's kappa: {cohen_kappa_score(pred_cluster, label_cluster)}")

        column_names = [f'Prob_{i + 1}' for i in range(class_num)]
        prob_df = pd.DataFrame(prob_cluster, columns=column_names)
        df = pd.DataFrame({'File Path': name_cluster, 'Label': label_cluster, 'Pred': pred_cluster})
        df = pd.concat([df, prob_df], axis=1)
        excelname = os.path.basename(args.folder_arbitrary_path)
        exceldir = os.path.join(model_path, '%s.xlsx' % excelname)
        df.to_excel(exceldir, index=False)
    else:
        folder_arbitrary_path_all = [
            # "../bs/33_18720/data_twodoctors.xlsx",
            # "../bs/33_18720/reader.xlsx",
            # "../bs/33_18720/test2_wu0.xlsx",
            # "../bs/33_18720/test4_2122.xlsx",
            "../bs/33_18720/test1_23.xlsx",
            # "../bs/test2_wu0_0.6.xlsx",
            # "../bs/33_18720/test2_wu0_0.9.xlsx",
            # "../bs/33_18720/reader_0.9.xlsx",
            # "../bs/33_18720/0.9test1_23.xlsx",
            # "../bs/1_9984/test1_23.xlsx",
            # "../bs/1_9984/test4_2122.xlsx",
        ]
        # class_num = 2
        for folder_arbitrary_path in folder_arbitrary_path_all:
            model_path = os.path.join(args.model_path, 'bs', args.ssl_weights_path.replace('/', '>'),
                                      # os.path.basename(args.folder_arbitrary_path)
                                      )
            os.makedirs(model_path, exist_ok=True)
            B = 1000
            df = pd.read_excel(folder_arbitrary_path)
            column_data = {}
            # 打印每一列数据
            for i, column in enumerate(df.columns):
                column_data[i] = df[column].tolist()
            sample_statistics = []  # 存储每次采样的统计量
            doctor_score = torch.tensor(column_data[2]) - 1
            # doctor_score = torch.where(doctor_score < 3, torch.tensor([0]), torch.tensor([1]))
            # doctor_score = torch.where(doctor_score > 0, torch.tensor([1]), torch.tensor([0]))
            sample_size = len(doctor_score)  # 样本数量保持和原数据一致
            label = torch.tensor(column_data[1]) - 1
            # 使用列表解析来收集第 3 到第 6 列的数据，然后合并成一个 Tensor
            if class_num == 4:
                prob = torch.tensor([column_data[i] for i in range(3, 7)]).T  # 转置为正确的形状
                for b in tqdm(range(B)):
                    random_indices = torch.randint(0, sample_size, (sample_size,))  # 生成随机索引
                    doctor_score_random = doctor_score[random_indices]  # 根据随机索引抽取样本
                    label_random = label[random_indices]  # 根据随机索引抽取样本

                    prob_random = prob[random_indices]  # 根据随机索引抽取样本
                    roc = AUROC(task="multiclass", num_classes=class_num, average=None)(prob_random, label_random)
                    micro_roc_auc = roc_auc_score(label_random, prob_random, average='micro', multi_class='ovr')
                    eval_dict, k_acc, kappa = evaluate_confusion_metrics(doctor_score_random, label_random,
                                                                         class_num)

                    sample_statistic = collect_statistics(eval_dict, k_acc, micro_roc_auc, roc, kappa, class_num)
                    sample_statistics.append(sample_statistic)

                roc = AUROC(task="multiclass", num_classes=class_num, average=None)(prob, label)
                micro_roc_auc = roc_auc_score(label, prob, average='micro', multi_class='ovr')
                eval_dict, k_acc, kappa = evaluate_confusion_metrics(doctor_score, label, class_num)

                sample_statistic = collect_statistics(eval_dict, k_acc, micro_roc_auc, roc, kappa, class_num)
                sample_statistics.append(sample_statistic)

            elif class_num == 2:
                prob = torch.tensor([column_data[i] for i in range(3, 5)]).T  # 转置为正确的形状
                for b in tqdm(range(B)):
                    random_indices = torch.randint(0, sample_size, (sample_size,))  # 生成随机索引
                    doctor_score_random = doctor_score[random_indices]  # 根据随机索引抽取样本
                    label_random = label[random_indices]  # 根据随机索引抽取样本
                    eval_dict, k_acc, kappa = evaluate_confusion_metrics(doctor_score_random, label_random,
                                                                         class_num)

                    sample_statistic = collect_statistics_doctor(eval_dict, k_acc, kappa, class_num)
                    sample_statistics.append(sample_statistic)
                eval_dict, k_acc, kappa = evaluate_confusion_metrics(doctor_score, label, class_num)

                sample_statistic = collect_statistics_doctor(eval_dict, k_acc, kappa, class_num)
                sample_statistics.append(sample_statistic)

            confidence_level = 0.95
            confidence_intervals = calculate_confidence_intervals(sample_statistics, confidence_level)
            filename = os.path.basename(folder_arbitrary_path).split(".")[-2]
            file_path = os.path.join(model_path, f'{filename}.txt')
            excel_path = os.path.join(model_path, f'{filename}.xlsx')
            write_statistics(file_path, excel_path, confidence_intervals, class_num, B, args)

    end_time = time.time()
    print('Time:{:.1f}s'.format(end_time - start_time))


def docter(args, class_num=4, bs=False, ):
    # 读取Excel文件
    excel_file = './reader1.xlsx'  # 将文件路径替换为您的Excel文件路径
    df = pd.read_excel(excel_file)
    column_data = {}
    # 打印每一列数据
    for i, column in enumerate(df.columns):
        column_data[i] = df[column].tolist()

    model_path = os.path.join(args.model_path, 'testchort2', 'docter')
    os.makedirs(model_path, exist_ok=True)

    class_num = 2
    recall_list, specificity_list, ppv_list, npv_list, f1_list, kacc_list, doctor_list = [], [], [], [], [], [], []
    kappa_list = []
    ave_confusion_matrix = torch.zeros((class_num, class_num))

    if class_num == 4:
        label = torch.tensor(column_data[1]) - 1
        eval_ind_model = evaluating_indicator(class_num)
        eval_dict_model = eval_ind_model.summerize_confusion_matrix()

        # for i in range(10):
        for i in [0, 2, 3, 4, 5, 6, 1, 7, 8, 9]:
            B = 1000 if bs else 1
            sample_statistics = []  # 存储每次采样的统计量
            doctor_score = torch.tensor(column_data[i + 2]) - 1
            sample_size = len(doctor_score)  # 样本数量保持和原数据一致
            for b in tqdm(range(B)):
                # for i in [0, 2, 3, 4, 5, 6]:
                # for i in [1,7]:
                # for i in [8, 9]:
                random_indices = torch.randint(0, sample_size, (sample_size,))  # 生成随机索引
                doctor_score_random = doctor_score[random_indices]  # 根据随机索引抽取样本
                label_random = label[random_indices]  # 根据随机索引抽取样本
                eval_dict, k_acc, kappa = evaluate_confusion_metrics(doctor_score_random, label_random, class_num)

                sample_statistic = collect_statistics_doctor(eval_dict, k_acc, kappa, class_num)
                sample_statistics.append(sample_statistic)

            eval_dict, k_acc, kappa = evaluate_confusion_metrics(doctor_score, label, class_num)
            # eval_ind = evaluating_indicator(class_num)
            # eval_ind.update_confusion_matrix(doctor_score, label)
            # eval_dict = eval_ind.summerize_confusion_matrix()
            # ave_confusion_matrix += eval_ind.confusion_matrix / 10

            sample_statistic = collect_statistics_doctor(eval_dict, k_acc, kappa, class_num)
            sample_statistics.append(sample_statistic)

            if not bs:
                doctor_list.append(doctor_score.numpy())
                # Precision NPV
                recall_list.append(eval_dict['Recall'])
                specificity_list.append(eval_dict['Specificity'])
                ppv_list.append(eval_dict['Precision'])
                npv_list.append(eval_dict['NPV'])
                f1_list.append(eval_dict['F1'])
                kacc_list.append(k_acc)
                kappa_list.append(kappa)
            else:
                confidence_level = 0.95
                confidence_intervals = calculate_confidence_intervals(sample_statistics, confidence_level)
                file_path = os.path.join(model_path, f'++doctor{i + 1}.txt')
                excel_path = os.path.join(model_path, f'++doctor{i + 1}.xlsx')
                write_statistics(file_path, excel_path, confidence_intervals, class_num, B, args)
    if not bs:
        doctor_numpy = np.array(doctor_list)
        kappa = fleiss_kappa1(doctor_numpy)
        print(f"Fleiss' Kappa: {kappa}")

        # 医生Cohen's Kappa图
        kappa_path = os.path.join(model_path, '+Kappa.png')
        plot_cohens_kappa_and_metrics(doctor_numpy, kappa_path, )

        # 医生表格
        recall_tensor, specificity_tensor = torch.stack(recall_list), torch.stack(specificity_list)
        ppv_tensor, npv_tensor = torch.stack(ppv_list), torch.stack(npv_list)
        f1_tensor = torch.stack(f1_list)
        kacc_tensor = torch.stack(kacc_list).unsqueeze(1)
        kappa_tensor = torch.stack(kappa_list).unsqueeze(1)
        data_tensor = torch.cat(
            [recall_tensor, specificity_tensor, ppv_tensor, npv_tensor, f1_tensor, kacc_tensor, kappa_tensor],
            dim=1)
        df = pd.DataFrame(data_tensor)
        # 保存到 Excel 文件
        # excelname = os.path.basename(args.folder_arbitrary_path)
        excelname = 'doctor_Recall+Specificity'
        exceldir = os.path.join(model_path, '%s.xlsx' % excelname)
        df.to_excel(exceldir, index=False, header='recall')  # index=False 用来避免保存时生成额外的索引列

        # Plot confusion matrix for eval_ind
        confusion_matrix_path = os.path.join(model_path, '+ConfusionMatrix_doctor.png')
        plot_confusion_matrix(ave_confusion_matrix, confusion_matrix_path)
        # plot_confusion_matrix(eval_ind.confusion_matrix, confusion_matrix_path)

    file_path = os.path.join(model_path, '++pred_classify.txt')
    with open(file_path, 'a', encoding='utf-8') as f:
        # f.write('\nTest dataset. Path: %s\n' % args.folder_arbitrary_path)
        f.write('\nTest dataset. Path: %s\n' % excel_file)
        f.write(f'Loaded from docter\n')
        for key, value in eval_dict.items():
            # value = value.item()
            # value = torch.round(value, decimals=3)
            f.write(f"{key}: {value}\n")
        f.write(f"k_acc: {k_acc}\n")


def plot_confusion_matrix(conf_matrix, confusion_matrix_path):
    plt.figure()
    precision_ax = sns.heatmap(conf_matrix, annot=True, fmt='.0f')
    precision_ax.set_title('Confusion Matrix')
    precision_ax.set_xlabel('Predicted Label')
    precision_ax.set_ylabel('True Label')
    precision_ax.set_xticklabels(range(1, conf_matrix.shape[1] + 1))
    precision_ax.set_yticklabels(range(1, conf_matrix.shape[0] + 1))
    plt.savefig(confusion_matrix_path)
    plt.show()


def plot_cohens_kappa_and_metrics(doctor_numpy, kappa_path, ):
    # 医生Cohen's Kappa图
    num_observers = doctor_numpy.shape[0]
    kappa_matrix = np.ones((num_observers, num_observers))

    for i in range(num_observers):
        for j in range(i + 1, num_observers):
            kappa_value = cohen_kappa_score(doctor_numpy[i], doctor_numpy[j])
            kappa_matrix[i, j] = kappa_value
            kappa_matrix[j, i] = kappa_value  # 对称矩阵

    # 使用 seaborn 绘制热图
    sns.heatmap(kappa_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1,
                xticklabels=np.arange(1, num_observers + 1),
                yticklabels=np.arange(1, num_observers + 1))
    plt.title('Inter-observer Agreement (Cohen\'s Kappa)')
    plt.xlabel('Observer')
    plt.ylabel('Observer')
    plt.savefig(kappa_path)
    plt.show()


def collect_statistics(eval_dict, k_acc, micro_roc_auc, roc, kappa, class_num):
    if class_num == 2:
        return (
            eval_dict['Recall'][-1],
            eval_dict['Specificity'][-1],
            eval_dict['Precision'][-1],
            eval_dict['NPV'][-1],
        )
    else:
        return (
            k_acc,
            kappa,
            micro_roc_auc,
            *roc,
            *eval_dict['Recall'],
            *eval_dict['Specificity'],
            eval_dict['Precision'][-1],
            eval_dict['NPV'][-1],
            *eval_dict['F1'],
        )


def collect_statistics_doctor(eval_dict, k_acc, kappa, class_num):
    if class_num == 2:
        return (
            eval_dict['Recall'][-1],
            eval_dict['Specificity'][-1],
            eval_dict['Precision'][-1],
            eval_dict['NPV'][-1],
        )
    else:
        return (
            k_acc,
            kappa,
            # record.micro_roc_auc,
            # *record.roc_auc,
            *eval_dict['Recall'],
            *eval_dict['Specificity'],
            eval_dict['Precision'][-1],
            eval_dict['NPV'][-1],
            *eval_dict['F1'],
        )


def calculate_confidence_intervals(sample_statistics, confidence_level):
    confidence_intervals = []
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    x = sample_statistics[-1]
    data_for_statistics = sample_statistics[:-1]

    for i in range(len(data_for_statistics[0])):
        statistics = [sample[i] for sample in data_for_statistics]
        statistics = torch.Tensor(statistics).detach().cpu().numpy()
        lower_bound = np.percentile(statistics, lower_percentile)
        upper_bound = np.percentile(statistics, upper_percentile)
        confidence_intervals.append((x[i], lower_bound, upper_bound))

    return confidence_intervals


def write_statistics(file_path, excel_path, confidence_intervals, class_num, B, args):
    if class_num == 2:
        metrics = ['Sen', 'Spec', 'PPV', 'NPV', ]
    else:
        metrics = ['k_acc',
                   'kappa',
                   'micro_roc_auc',
                   'auc1',
                   'auc2', 'auc3', 'auc4',
                   'sen1',
                   'sen2', 'sen3', 'sen4',
                   'spec1',
                   'spec2', 'spec3', 'spec4',
                   'PPV', 'NPV',
                   'F1-1',
                   'F1-2', 'F1-3', 'F1-4',
                   ]
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f'\nTest dataset. Path: {args.folder_arbitrary_path}\n')
        f.write(f'Class number: {args.class_num}\n')
        f.write(f'Bootstraps: {B}\n')
        if args.load_ssl_weights:
            f.write(f'Loaded from checkpoint. Path: {args.ssl_weights_path}\n')
        else:
            f.write(f'Loaded from checkpoint. Path: {args.model_weights_path}/epoch_dncnn_{args.load_step}\n')

        for i, (x, a, b) in enumerate(confidence_intervals):
            x = x.item() if hasattr(x, 'item') else x
            lower = a.item() if hasattr(a, 'item') else a
            upper = b.item() if hasattr(b, 'item') else b
            f.write(f"{metrics[i]}: {x:.3f}\t{lower:.3f}\t{upper:.3f}\n")

        # 生成表格数据
    data = []
    for i, (x, a, b) in enumerate(confidence_intervals):
        x = x.item() if hasattr(x, 'item') else x
        lower = a.item() if hasattr(a, 'item') else a
        upper = b.item() if hasattr(b, 'item') else b
        data.append([metrics[i], x, lower, upper])

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=["Metric", "Value", "Lower CI", "Upper CI"])

    # 保存到 Excel 文件
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        # 在同一个文件中添加新工作表
        df.to_excel(writer, sheet_name="Statistics", index=False)


def evaluate_confusion_metrics(doctor_score, label, class_num):
    # 初始化评估指标对象
    eval_ind = evaluating_indicator(class_num)

    # 更新混淆矩阵
    eval_ind.update_confusion_matrix(doctor_score, label)

    # 计算微观准确率（k_acc）
    metric = MulticlassAccuracy(average="micro", num_classes=class_num)
    k_acc = metric(doctor_score, label)

    # 获取混淆矩阵汇总
    eval_dict = eval_ind.summerize_confusion_matrix()

    # 计算Cohen's Kappa
    kappa = torch.tensor(cohen_kappa_score(doctor_score.numpy(), label.numpy()))

    return eval_dict, k_acc, kappa


def fleiss_kappa(data: np.array):
    """
    Calculates Fleiss' kappa coefficient for inter-rater agreement.
    Args:
        data: numpy array of shape (subjects, categories), where each element represents
              the number of raters who assigned a particular category to a subject.
    Returns:
        kappa: Fleiss' kappa coefficient.
    """
    subjects, categories = data.shape
    n_raters = np.sum(data[0])  # Total number of raters

    # Step 1: Calculate P_i for each subject
    P_i = (np.sum(data ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Step 2: Calculate P_bar (average of P_i across all subjects)
    P_bar = np.mean(P_i)

    # Step 3: Calculate P_j (proportion of all assignments which were to category j)
    P_j = np.sum(data, axis=0) / (subjects * n_raters)

    # Step 4: Calculate P_e_bar (expected proportion of agreement)
    P_e_bar = np.sum(P_j ** 2)

    # Step 5: Calculate kappa
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


def fleiss_kappa1(data: np.array):
    """
    Calculates Fleiss' kappa coefficient for inter-rater agreement.
    Args:
        data: numpy array of shape (subjects, categories), where each element represents
              the number of raters who assigned a particular category to a subject.
    Returns:
        kappa: Fleiss' kappa coefficient.
    """

    # ratings = np.array(data)
    # Number of subjects (projects in this case)
    subjects = data.shape[1]
    # Number of categories (unique ratings)
    categories = np.max(data) + 1  # Assuming ratings are from 0 to 1
    # Initialize the input matrix for Fleiss' Kappa
    input_matrix = np.zeros((subjects, categories), dtype=int)
    # Populate the input matrix based on the ratings
    for j in range(subjects):
        for i in range(data.shape[0]):
            rating = data[i, j]
            input_matrix[j, rating] += 1
    data = input_matrix

    subjects, categories = data.shape
    n_rater = np.sum(data[0])

    p_j = np.sum(data, axis=0) / (n_rater * subjects)
    P_e_bar = np.sum(p_j ** 2)

    P_i = (np.sum(data ** 2, axis=1) - n_rater) / (n_rater * (n_rater - 1))
    P_bar = np.mean(P_i)

    K = (P_bar - P_e_bar) / (1 - P_e_bar)

    return K


def clear_all_gpu_memory():
    # 删除模型和数据
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            del obj
