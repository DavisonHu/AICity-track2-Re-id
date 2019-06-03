from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from PIL import Image
from opt import opt
import glob
import pandas as pd
import numpy as np
import os.path as osp


class Data(object):
    def __init__(self):
        # paper is (384, 128)
        train_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = AICityCar(train_transform, 'train', opt.data_path)
        self.testset = AICityCar(test_transform, 'test', opt.data_path)
        self.queryset = AICityCar(test_transform, 'query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=1, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=1,
                                                  pin_memory=True)


def blank_extention(ori_image):
    blanksize = max(ori_image.size[0], ori_image.size[1])
    blank = Image.new("RGB", (blanksize, blanksize), (255, 255, 255))
    if ori_image.size[0] > ori_image.size[1]:
        blank.paste(ori_image, (0, int(blanksize / 2) - int(min(ori_image.size[0], ori_image.size[1]) / 2)))
    else:
        blank.paste(ori_image, (int(blanksize / 2) - int(min(ori_image.size[0], ori_image.size[1]) / 2), 0))
    return blank


def process_dir(dir_path, data):
    id_list = np.array(pd.read_csv(dir_path + '/label.csv', header=None))
    id = []
    color = []
    if dir_path[-1] == 'n':
        for num, (_) in enumerate(data):
            if int(id_list[num][0]) <= 95:
                pid = int(id_list[num][0]) - 1
            else:
                pid = int(id_list[num][0]) - 146
            id.append(pid)
            color.append(int(id_list[num][3])-1)
    else:
        for num, (_) in enumerate(data):
            id.append(int(id_list[num][0]))
            color.append(int(id_list[num][0]))
    return np.array(id), np.array(color)


def find_data_dir(dir_path, image_num):
    id_list = np.array(pd.read_csv(dir_path + '/label.csv', header=None))
    data_dir = []
    for num in range(image_num):
        data_dir.append(dir_path + '/' + str(id_list[num][1]))
    return data_dir


class AICityCar(dataset.Dataset):
    """
        For CVPR 2019 AICity Challenge track 2
        https://www.aicitychallenge.org/

        Notice:Because its for competition, query image ID is unknown
        """
    def __init__(self, transform, dtype, data_path):
        super(AICityCar, self).__init__()
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        if dtype == 'train':
            self.data_path += '/image_train'
        elif dtype == 'test':
            self.data_path += '/image_test'
        else:
            self.data_path += '/image_query'

        self.imgs_num = len(glob.glob(osp.join(self.data_path, '*.jpg')))
        self.imgs = find_data_dir(self.data_path, self.imgs_num)
        self.target, self.color = process_dir(self.data_path, self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.target[index]
        color = self.color[index]
        img = blank_extention(self.loader(path))
        if self.transform is not None:
            img = self.transform(img)

        return img, target, color

    def __len__(self):
        return len(self.imgs)
