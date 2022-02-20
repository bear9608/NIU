import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('save-feature-model')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self,data):
        data_shape = data.shape[:-3]
        img_shape = data.shape[-3:]
        data = data.view(-1,*img_shape)
        x_data = self.encoder(data)
        x_data = x_data.view(*data_shape,-1)


        return x_data