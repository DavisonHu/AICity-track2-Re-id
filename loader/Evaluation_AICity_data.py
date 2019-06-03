from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
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

        self.trainset = EvaluationAICityCar(train_transform, 'train', opt.data_path)
        self.testset = EvaluationAICityCar(test_transform, 'test', opt.data_path)
        self.queryset = EvaluationAICityCar(test_transform, 'query', opt.data_path)
        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=1, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=1,
                                                  pin_memory=True)


def process(data_path, dtype, num):
    img = []
    car_id = []
    id_list = np.array(pd.read_csv(data_path + '/train_label.csv', header=None))
    if dtype == 'train':
        for i in range(10):
            if i != num:
                path = data_path + '/folder' + str(i) + '/'
                dir_path = glob.glob(osp.join(path, '*.jpg'))
                for j in range(len(dir_path)):
                    img.append(dir_path[j])
                    tmp = str(dir_path[j])
                    car_id.append(id_list[int(tmp[-10:-4])-1][0])
    elif dtype == 'test':
        path = data_path + '/folder' + str(num) + '/test.txt'
        data = np.array(pd.read_csv(path, header=None))
        for i in range(data.shape[0]):
            tmp = data_path + '/folder' + str(num) + '/' + data[i]
            img.append(tmp[0])
            tmp = str(data[i])
            car_id.append(id_list[int(tmp[-12:-6]) - 1][0])
    elif dtype == 'query':
        path = data_path + '/folder' + str(num) + '/query.txt'
        data = np.array(pd.read_csv(path, header=None))
        for i in range(data.shape[0]):
            tmp = data_path + '/folder' + str(num) + '/' + data[i]
            img.append(tmp[0])
            tmp = str(data[i])
            car_id.append(id_list[int(tmp[-12:-6]) - 1][0])

    return img, car_id


class EvaluationAICityCar(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        super(EvaluationAICityCar, self).__init__()
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        self.imgs, self.id = process(self.data_path, dtype, num=6)

    def __getitem__(self, index):
        path = self.imgs[index]
        if self.id[index] <= 95:
            target = self.id[index] - 1
        else:
            target = self.id[index] - 146
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
