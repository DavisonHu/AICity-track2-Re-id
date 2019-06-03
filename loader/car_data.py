from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import random
import glob
from PIL import Image
import numpy as np


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

        self.trainset = Car(transforms=train_transform, root=opt.data_path)
        self.testset = Car(transforms=test_transform, root=opt.data_path)
        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=1, pin_memory=True)


def blank_extention(ori_image):
    blanksize = max(ori_image.size[0], ori_image.size[1])
    blank = Image.new("RGB", (blanksize, blanksize), (255, 255, 255))
    if ori_image.size[0] > ori_image.size[1]:
        blank.paste(ori_image, (0, int(blanksize / 2) - int(min(ori_image.size[0], ori_image.size[1]) / 2)))
    else:
        blank.paste(ori_image, (int(blanksize / 2) - int(min(ori_image.size[0], ori_image.size[1]) / 2), 0))
    return blank


def find_car_id(dir):
    car_id = 0
    num = 1
    for i in range(-2, -7, -1):
        if dir[i] != '/':
            car_id += int(dir[i])*num
            num = num*10
        else:
            break
    return car_id-1


class Car(dataset.Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.files_A = sorted(glob.glob(root + '/*/'))

    def __getitem__(self, index):
        image_dir = glob.glob(self.files_A[index] + '*.*')
        car_side = np.array(random.randint(0, 4))
        car_id = find_car_id(self.files_A[index])
        return self.transform(blank_extention(default_loader(image_dir[car_side]))), car_id, car_side

    def __len__(self):
        return len(self.files_A)
