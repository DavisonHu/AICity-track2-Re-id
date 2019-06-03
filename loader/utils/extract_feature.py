import torch
import numpy as np


def extract_feature(model, loader):
    features = torch.FloatTensor()
    for batch, (inputs, id_label, color_label) in enumerate(loader):
        inputs = inputs.to('cuda')
        output = model(inputs)[0].cpu()
        if batch == 0:
            features = output
        else:
            features = np.append(features, output, axis=0)
    return features
