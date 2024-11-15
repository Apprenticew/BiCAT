import os
import json
import argparse
import pprint
import torch
import torch.nn as nn
import torch.nn.init as init
from model import DnCNN
from model import kaiming_normal_weights_init
from resnet import resnet18
from trainer import trainer
from tester import tester
from data import train_dl, arbitraty_dl

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--model_path', default='./result/')  # 保存路径
parser.add_argument('--folder_path_gray', default='./data1/Retrospective_train_cohort')
parser.add_argument('--folder_test_path_gray', default='./data1/Retrospective_test_cohort')
parser.add_argument('--folder_unlabel_path', default='../unlabeled/')
parser.add_argument('--folder_arbitrary_path', default='./data1/Prospective_test_cohort')
parser.add_argument('--load_ssl_weights', action='store_true', default=False)
parser.add_argument('--ssl_weights_path', default='./data/curr_iter_18720')

parser.add_argument('--bsize', type=int, default=16)
parser.add_argument('--num_iters', type=int, default=20000)
parser.add_argument('--class_num', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--u_ratio', type=int, default=7)
parser.add_argument('--u_data_ratio', type=int, default=7)

args = parser.parse_args()


# 保存到json
def save():
    # parser.add_argument('--epoch', type=int, default=100, help='epoch')
    # args = parser.parse_args()
    if args.test:
        path = os.path.join(args.model_path, args.ssl_weights_path.replace('/', '>'),
                            os.path.basename(args.folder_arbitrary_path))
        os.makedirs(path, exist_ok=True)
    else:
        path = args.model_path
    with open(os.path.join(path, '+model_arg.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)  # indent意思就是json格式缩进4个space，便于肉眼查看
        # dump()方法的第一个参数是dict，第二个参数是打开的文件句柄，第三个参数是缩进的位数


# 加载json
def load():
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)  # {}
    with open(os.path.join(args.model_load_path, 'model_arg.json'), 'rt') as f:
        args_dict.update(json.load(f))
    print(args_dict)  # {'seed_1': 98, 'epoch': 100}


class Model(nn.Module):
    def __init__(self, model1, model2):
        super(Model, self).__init__()
        self.dncnn = model1
        self.resnet = model2
        # shortcut
        self.shortcut = nn.Sequential()

    def forward(self, img):
        mid = self.dncnn(img)
        img = self.shortcut(img)
        score = self.resnet(mid + img)
        return score


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pprint.pprint(vars(args))

    # 初始化网络和损失函数
    dncnn = DnCNN(depth=7, image_channels=1, out_channels=32).to(device)
    resnet = resnet18(args.class_num).to(device)

    print('\nInitialized checkpoint parameters..')
    dncnn.apply(kaiming_normal_weights_init)
    resnet.apply(kaiming_normal_weights_init)

    model = Model(dncnn, resnet).to(device)

    # total_params = sum(p.numel() for p in model.parameters())  # 计算总参数数量
    # print(f"Total parameters: {total_params}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # summary(model, input_size=[(1, 128, 128), (1, 1, 1)])
    # target_layers = [model.resnet.conv5_x[-1]]
    # print(target_layers)

    if not args.test:
        train_dataloader, unlabel_dataloader, test_dataloader, class_counts = train_dl(args)
        trainer(model, train_dataloader, unlabel_dataloader, test_dataloader, class_counts, args)

    else:
        have_lable = True
        class_num = args.class_num
        if args.folder_arbitrary_path == "./data/External_multi_centers/ALL":
            class_num = 3
        bs = False
        heatmap = True
        mode = 'liberal'
        arbitraty_dataloader, class_counts = arbitraty_dl(args, have_lable=have_lable, class_num=class_num, bs=bs)
        # docter(args, class_num, bs=bs, )
        tester(model, arbitraty_dataloader, args, class_counts, class_num=class_num, bs=bs, heatmap=heatmap, mode=mode)


if __name__ == '__main__':
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)
    # load()
    save()
    main(args)


def load_path(folder_path):
    folder_path_gray = {}
    categories = 4
    if categories == 4:
        for q in range(categories):
            folder_path_cluster = []
            folder_path = os.path.join(folder_path, "%s" % str(q + 1))
            for file_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file_name)
                folder_path_cluster.append(image_path)
            folder_path_gray[q] = folder_path_cluster
    data = [(k, v) for k, vs in folder_path_gray.items() for v in vs]

    return data
