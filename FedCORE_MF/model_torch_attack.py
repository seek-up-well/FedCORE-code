import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from const import *


class MF_attack(nn.Module):
    def __init__(self, maxn, maxm, hidden_dim, dropout):
        super(MF_attack, self).__init__()
        self.user_size = maxn + 3
        self.item_size = maxm + 3
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.uembedding_layer = nn.Embedding(self.user_size, hidden_dim)
        self.uembedding_layer.weight.data.copy_(
            (torch.rand_like(self.uembedding_layer.weight.data) - 0.5)*0.01)
        self.uembedding_layer.weight.data.requires_grad = True

    def forward(self, userid_input, iemb):

        uemb = self.uembedding_layer(userid_input)

        iemb = iemb

        uemb = uemb.squeeze(1).unsqueeze(2)
 
        logger.debug(f"iemb_h size: {iemb.size()}")
        logger.debug(f"uem_h size: {uemb.size()}")

        pred = torch.bmm(iemb, uemb).squeeze(2)
        logger.debug(f"pred size: {pred.size()}")
        return pred

    def load_parameters(self, params):
        for param, sparam in zip(params, self.parameters()):
            sparam.data.copy_(param.data)
