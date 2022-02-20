import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import utils
from .models import register


@register('meta-baseline-demo')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.fc = torch.nn.Linear(10,5)
        # self.fc = torch.nn.Linear(10, 5)
        # self.fc1 = torch.nn.Linear(10,5)
        # self.fc2 = torch.nn.Linear(10,5)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        for name, p in self.named_parameters():
                if name == 'fc1.weight':
                    p.requires_grad = True
                elif name == 'fc1.bias':
                    p.requires_grad = True
                elif name == 'fc2.weight':
                        p.requires_grad = True
                elif name == 'fc2.bias':
                        p.requires_grad = True
                if name == 'fc.weight':
                        p.requires_grad = True
                elif name == 'fc.bias':
                        p.requires_grad = True
                else:
                    p.requires_grad = False
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp


    def forward(self, x_shot, x_query, x_data):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)


        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            x_data = F.normalize(x_data, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits_query = utils.compute_logits(
                x_query, x_data, metric=metric)
        logits_shot = utils.compute_logits(
                x_shot, x_data, metric=metric)
        logits_demo = utils.compute_logits(
                logits_query, logits_shot, metric='delete',temp= self.temp)
        logits = utils.compute_logits(
                x_query, x_shot, metric=metric,temp= self.temp)
        logits_add = utils.compute_logits(
                x_query, x_shot, metric='add',temp= self.temp)

        logits = F.softmax(logits,dim=1)
        logits_demo = F.softmax(logits_demo,dim =1)
        logits_add = F.softmax(logits_add,dim =1)
        logits = logits*self.temp
        logits_demo = logits_demo*self.temp
        logits_add = logits_add*self.temp
        # logitsfc = torch.cat((logits_add,logits_demo,logits_add),2)
        logitsfc = torch.cat((logits_add, logits_demo), 2)
        # x = self.fc(logitsfc)
        x = logits_demo
        # logitsfc = torch.cat((logits,logits_demo),2)
        # x = self.fc(logitsfc)
        # x = logits_demo
        return logits,x,logits_add

