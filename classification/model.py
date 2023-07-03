import torch
import torch.nn as nn

class ViT_MCMC(nn.Module):
    def __init__(self,pretrained_model,num_classes):
        super(ViT_MCMC, self).__init__()
        self.pretrained_model = pretrained_model
        self.FC = nn.Linear(1000,num_classes)
    def forward(self,x):
        x = self.pretrained_model(x)
        x = self.FC(x)
        return x