import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import visdom
from torch.optim import lr_scheduler
from opt import opt
from loader import AICity_data
from model.pyramid_resnet import Model
from loss.loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import re_ranking, mean_ap
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
vis = visdom.Visdom(env='pyramid-v5-RE-ID', port=8098)


class Main(object):

    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        self.model.train()
        tri = 0
        id = 0
        color = 0
        center = 0
        id1 = 0
        id2 = 0
        id3 = 0
        color1 = 0
        color2 = 0
        color3 = 0
        total = 0
        for batch, (inputs, id_label, color_label) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            id_label = id_label.to('cuda')
            color_label = color_label.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss, tri_loss, id_loss, color_loss, center_loss = self.loss(outputs, id_label, color_label)
            loss.backward()
            self.optimizer.step()

            tri += tri_loss.item()
            id += id_loss.item()
            color += color_loss.item()
            center += center_loss.item()

            total += inputs.size(0)
            _, id1_predicted = torch.max(outputs[4].data, 1)
            _, id2_predicted = torch.max(outputs[5].data, 1)
            _, id3_predicted = torch.max(outputs[6].data, 1)
            _, color1_predicted = torch.max(outputs[7].data, 1)
            _, color2_predicted = torch.max(outputs[8].data, 1)
            _, color3_predicted = torch.max(outputs[9].data, 1)

            id1 += (id1_predicted == id_label).sum().item()
            id2 += (id2_predicted == id_label).sum().item()
            id3 += (id3_predicted == id_label).sum().item()
            color1 += (color1_predicted == color_label).sum().item()
            color2 += (color2_predicted == color_label).sum().item()
            color3 += (color3_predicted == color_label).sum().item()

        self.scheduler.step()

        return np.array(tri/len(self.train_loader)), np.array(id/len(self.train_loader)), \
               np.array(color/len(self.train_loader)), np.array(center/len(self.train_loader)), np.array(id1/total), \
               np.array(id2/total), np.array(id3/total), np.array(color1/total), np.array(color2/total), \
               np.array(color3/total),

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        queryset = self.queryset
        testset = self.testset
        m_ap = mean_ap(dist, queryset, testset)

        print('[With    Re-Ranking] mAP: {:.4f}'.format(m_ap))

        #########################no re rank##########################
        dist = cdist(qf, gf)
        m_ap = mean_ap(dist, queryset, testset)

        print('[Without Re-Ranking] mAP: {:.4f}'.format(m_ap))

    def evaluate_ai(self):
        self.model.load_state_dict(torch.load('weights/AI_mgn/pyramid_v4.pth'))
        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        for i in range(dist.shape[0]):
            txt_list = np.zeros((100))
            for j in range(txt_list.shape[0]):
                txt_list[j] = int(np.argmin(dist[i])) + 1
                dist[i][int(txt_list[j]) - 1] = 100000
            txt_list = pd.DataFrame(txt_list, dtype=np.int32)
            txt_list.to_csv('save/{:0>6d}.txt'.format(i + 1), header=0, index=0)


def main():
    model = Model()

    pretrained_dict = torch.load('weights/AI_mgn/VERI.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    data = AICity_data.Data()
    loss = Loss()
    main = Main(model, loss, data)
    #main.evaluate_ai()
    if opt.mode == 'train':
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            outputs = main.train()
            vis.line(
                Y=np.column_stack((outputs[0], outputs[1], outputs[2], outputs[3])),
                X=np.column_stack((epoch, epoch, epoch, epoch)),
                win='Learning curve',
                update='append',
                opts={
                    'title': 'Learning curve',
                }
            )
            vis.line(
                Y=np.column_stack((outputs[4], outputs[5], outputs[6], outputs[7], outputs[8], outputs[9])),
                X=np.column_stack((epoch, epoch, epoch, epoch, epoch, epoch)),
                win='accuracy curve',
                update='append',
                opts={
                    'title': 'accuracy curve',
                }
            )
            if epoch % 10 == 0:
                print('\nstart evaluate')
                os.makedirs('weights/AI_mgn', exist_ok=True)
                torch.save(model.state_dict(), ('weights/AI_mgn/modelv5_{}.pth'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'aicity':
        print('start output txt files')
        model.load_state_dict(torch.load(opt.weight))
        main.AICity_evaluate()


if __name__ == '__main__':
    main()
