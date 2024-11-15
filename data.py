import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
import torchvision.transforms as transforms
from randaugment import RandAugment
from PIL import Image


class UnlabelDataset(Dataset):
    def __init__(self, folder_unlabel_path, transform, ):
        super(UnlabelDataset, self).__init__()
        self.train_unlabel_transform = transform
        self.data = self.load_data(folder_unlabel_path)
        self.rand_transform = RandAugment(4, v=0.1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        path = self.data[index]

        image = Image.open(path).convert('L')
        image = image.crop((96, 21, 448, 373))
        # image_w = self.weak_transform(image)
        # image_s = self.strong_transform(image)
        image_1 = self.rand_transform(image)
        image_2 = self.rand_transform(image_1)
        image_w = self.train_unlabel_transform(image_1)
        image_s = self.train_unlabel_transform(image_2)
        return image_w, image_s

    def load_data(self, folder_path):
        folder_path_cluster = []

        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                image_path = os.path.join(root, file_name)
                folder_path_cluster.append(image_path)

        return folder_path_cluster


class ArbitraryDataset(Dataset):
    def __init__(self, folder_path_gray, test_transform, have_lable=True, class_num=4, ):
        super(ArbitraryDataset, self).__init__()
        self.test_transform = test_transform
        self.have_lable = have_lable
        if self.have_lable:
            self.categories = class_num
            self.folder_path_gray = {}
            if self.categories == 4:
                for q in range(self.categories):
                    self.folder_path_gray[q] = self.load_data(os.path.join(folder_path_gray, "%s" % str(q + 1)))
            elif self.categories == 2:
                # self.folder_path_gray[0] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(1))) +
                #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
                #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))))
                # self.folder_path_gray[1] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
                # self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(1)))
                # self.folder_path_gray[1] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
                #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))))
                self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(3)))
                self.folder_path_gray[1] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
                # self.folder_path_gray[1] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
                #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))) +
                #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(4))))
            elif self.categories == 3:
                # self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(1)))
                # self.folder_path_gray[1] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
                #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))))
                # self.folder_path_gray[2] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
                self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(2)))
                self.folder_path_gray[1] = self.load_data(os.path.join(folder_path_gray, "%s" % str(3)))
                self.folder_path_gray[2] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
            self.class_counts = torch.tensor([len(value) for key, value in self.folder_path_gray.items()])
            self.data = [(k, v) for k, vs in self.folder_path_gray.items() for v in vs]
        else:
            self.data = self.load_data(folder_path_gray)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        [class_label, path] = self.data[index]
        image = Image.open(path).convert('L')
        # image = image.crop((45, 100, 45 + 565, 100 + 400))
        # image = image.crop((96, 21, 448, 373))
        image = self.test_transform(image)

        return image, class_label, path

    def load_data(self, folder_path):
        folder_path_cluster = []
        # 递归遍历文件夹及其子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.png') or file_name.endswith('.jpg'):
                    image_path = os.path.join(root, file_name)
                    folder_path_cluster.append(image_path)
        return folder_path_cluster


class LableDataset(Dataset):
    def __init__(self, folder_path_gray, class_num, transform, ):
        super(LableDataset, self).__init__()
        self.categories = class_num
        self.folder_path_gray = {}
        if self.categories == 4:
            for q in range(self.categories):
                self.folder_path_gray[q] = self.load_data(os.path.join(folder_path_gray, "%s" % str(q + 1)))
        elif self.categories == 2:
            # self.folder_path_gray[0] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(1))) +
            #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
            #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))))
            # self.folder_path_gray[1] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
            self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(3)))
            self.folder_path_gray[1] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
            # self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(1)))
            # self.folder_path_gray[1] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
            #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))) +
            #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(4))))
        elif self.categories == 3:
            # self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(1)))
            # self.folder_path_gray[1] = (self.load_data(os.path.join(folder_path_gray, "%s" % str(2))) +
            #                             self.load_data(os.path.join(folder_path_gray, "%s" % str(3))))
            # self.folder_path_gray[2] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
            self.folder_path_gray[0] = self.load_data(os.path.join(folder_path_gray, "%s" % str(2)))
            self.folder_path_gray[1] = self.load_data(os.path.join(folder_path_gray, "%s" % str(3)))
            self.folder_path_gray[2] = self.load_data(os.path.join(folder_path_gray, "%s" % str(4)))
        elif self.categories == 6:
            for q in range(self.categories):
                self.folder_path_gray[q] = self.load_data(os.path.join(folder_path_gray, "%s" % str(q + 1)))
        # self.class_counts = [len(value) for key, value in self.folder_path_gray.items()]
        self.class_counts = torch.tensor([len(value) for key, value in self.folder_path_gray.items()])
        self.data = [(k, v) for k, vs in self.folder_path_gray.items() for v in vs]
        self.train_label_transform = transform

        self.rand_transform = RandAugment(4, v=0.1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        [class_label, path] = self.data[index]
        image = Image.open(path).convert('L')
        # image = image.crop((96, 21, 448, 373))
        # image = self.rand_transform(image)
        image = self.train_label_transform(image)
        return image, class_label

    def load_data(self, folder_path):
        folder_path_cluster = []
        # 递归遍历文件夹及其子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.png'):
                    image_path = os.path.join(root, file_name)
                    folder_path_cluster.append(image_path)

        return folder_path_cluster


def transformers():
    size = 128
    mean, std = 0.345, 0.145
    train_label_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        RandAugment(4, v=0.1),
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_unlabel_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_label_transform, train_unlabel_transform, test_transform


def train_dl(args, ):
    train_label_transform, train_unlabel_transform, test_transform = transformers()

    train_set = LableDataset(args.folder_path_gray, args.class_num, train_label_transform)
    test_set = LableDataset(args.folder_test_path_gray, args.class_num, test_transform)
    unlabel_set = UnlabelDataset(args.folder_unlabel_path, train_unlabel_transform)

    train_size = int(len(train_set))
    unlabel_size = int(len(unlabel_set))
    test_size = int(len(test_set))
    sampler = RandomSampler(train_set, num_samples=args.num_iters * args.bsize, replacement=True)
    batch_sampler = BatchSampler(sampler, batch_size=args.bsize, drop_last=True)
    train_dataloader = DataLoader(train_set, num_workers=8, batch_sampler=batch_sampler)

    u_ratio = args.u_ratio
    if unlabel_size < args.u_data_ratio * train_size:
        sampler_u = RandomSampler(unlabel_set, num_samples=args.num_iters * args.bsize * u_ratio, replacement=True)
    else:
        sampler_u = RandomSampler(range(args.u_data_ratio * train_size),
                                  num_samples=args.num_iters * args.bsize * u_ratio, replacement=True)
    batch_sampler_u = BatchSampler(sampler_u, batch_size=args.bsize * u_ratio, drop_last=True)
    unlabel_dataloader = DataLoader(unlabel_set, num_workers=8, batch_sampler=batch_sampler_u)

    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=8, drop_last=False)
    class_counts = [train_set.class_counts, test_set.class_counts]

    return train_dataloader, unlabel_dataloader, [test_dataloader, ], class_counts


def arbitraty_dl(args, have_lable=True, class_num=4, bs=False):
    train_weak_transform, train_strong_transform, test_transform = transformers()

    arbitraty_set = ArbitraryDataset(args.folder_arbitrary_path, test_transform, have_lable, class_num, )
    sampler = RandomSampler(arbitraty_set, replacement=True)
    class_counts = arbitraty_set.class_counts  # 每个类别的样本数量列表
    test_size = int(len(arbitraty_set))
    if bs:
        arbitraty_dataloader = DataLoader(arbitraty_set, batch_size=64, sampler=sampler, num_workers=8,
                                          drop_last=False)
    else:
        arbitraty_dataloader = DataLoader(arbitraty_set, batch_size=16, shuffle=True, num_workers=8, drop_last=False)

    return arbitraty_dataloader, class_counts
