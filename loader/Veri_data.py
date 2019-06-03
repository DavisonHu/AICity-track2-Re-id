from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import pandas as pd


class Data(object):
    def __init__(self):
        # paper is (384, 128)
        train_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Veri(train_transform, 'train', opt.data_path)
        #self.testset = Veri(test_transform, 'test', opt.data_path)
        #self.queryset = Veri(test_transform, 'query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        #self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=1, pin_memory=True)
        #self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=1,
         #                                         pin_memory=True)


def process(data_path, dtype):
    data_list = pd.read_csv(data_path + '/label.csv').values
    img = []
    car_id = []
    car_color = []
    if dtype == 'query':
        id_list = pd.read_csv(data_path[0:-11] + 'image_test/label.csv').values
        for i in range(data_list.shape[0]):
            img.append(data_path + '/' + str(data_list[i][0]))
            car_id.append(id_list[data_list[i][1]-1][1])
            car_color.append(id_list[data_list[i][1] - 1][2])
    else:
        for i in range(data_list.shape[0]):
            img.append(data_path + '/' + str(data_list[i][0]))
            car_id.append(data_list[i][1])
            car_color.append(int(data_list[i][2]) - 1)

    return img, car_id, car_color


class Veri(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        super(Veri, self).__init__()
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        if dtype == 'train':
            self.data_path += '/image_train'
        elif dtype == 'test':
            self.data_path += '/image_test'
        else:
            self.data_path += '/image_query'

        self.img, self.id, self.color = process(self.data_path, dtype)

    def __getitem__(self, index):
        path = self.img[index]
        target = self.id[index]
        color = self.color[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, color

    def __len__(self):
        return len(self.img)
