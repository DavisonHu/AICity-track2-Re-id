from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from utils.Centerloss import CenterLoss


class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, id_label, color_label):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)
        center_loss = CenterLoss(num_classes=333, feat_dim=512)

        Triplet_Loss = [triplet_loss(output, id_label) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        Centor_Loss = [center_loss(output, id_label) for output in outputs[1:4]]
        Centor_Loss = sum(Centor_Loss) / len(Centor_Loss) * 0.0005

        CrossEntropy_Loss_id = [cross_entropy_loss(output, id_label) for output in outputs[4:7]]
        CrossEntropy_Loss_id = sum(CrossEntropy_Loss_id) / len(CrossEntropy_Loss_id)

        CrossEntropy_Loss_color = [cross_entropy_loss(output, color_label) for output in outputs[7:]]
        CrossEntropy_Loss_color = sum(CrossEntropy_Loss_color) / len(CrossEntropy_Loss_color)

        loss_sum = Triplet_Loss + 2 * (CrossEntropy_Loss_id + CrossEntropy_Loss_color) + Centor_Loss

        return loss_sum, Triplet_Loss, CrossEntropy_Loss_id, CrossEntropy_Loss_color, Centor_Loss
