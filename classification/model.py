import timm
import torch.nn as nn
class ScalpModel_4head(nn.Module):
    def __init__(self,pretrained_model):
        super(ScalpModel_4head, self).__init__()
        self.model = timm.create_model(pretrained_model, pretrained=True,  num_classes=1000)
        self.classifier1 = nn.Linear(1000, 3)
        self.classifier_dand = nn.Linear(1000, 4)
        self.classifier_seb = nn.Linear(1000, 4)
        self.classifier_ery = nn.Linear(1000, 4)

        # self.batch_norm1 = nn.BatchNorm1d(3)
        # self.batch_norm_dand = nn.BatchNorm1d(4)
        # self.batch_norm_seb = nn.BatchNorm1d(4)
        # self.batch_norm_ery = nn.BatchNorm1d(4)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.model(x)
        y1 = self.batch_norm1(self.classifier1(x))
        dand = self.classifier_dand(x)
        seb = self.classifier_seb(x)
        ery = self.classifier_ery(x)
        # dand = self.batch_norm_dand(self.classifier_dand(x))
        # seb = self.batch_norm_seb(self.classifier_seb(x))
        # ery = self.batch_norm_ery(self.classifier_ery(x))

        return self.sigmoid(y1),self.softmax(dand),self.softmax(seb),self.softmax(ery)


