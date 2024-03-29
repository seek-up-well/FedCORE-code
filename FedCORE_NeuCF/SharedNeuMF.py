import torch
from torch import nn
from torch.nn import functional as F


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, lamb=0.01, dropout=0.0, init=None):
        super(NeuMF, self).__init__()
        self.num_users = num_users+3
        self.num_items = num_items+3
        self.embed_dim = hidden_dim

        self.lamb = lamb

        self.dropout = nn.Dropout(p=dropout)

        self.gmf_user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        self.gmf_item_embedding = nn.Embedding(self.num_items, self.embed_dim)
        self.ncf_user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        self.ncf_item_embedding = nn.Embedding(self.num_items, self.embed_dim)
        self.hidden_layer1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.output_layer = nn.Linear(
            self.embed_dim + self.embed_dim, 1, bias=False)

        if init is not None:
            self.init_embedding(init)
        else:
            self.init_embedding(0)
 
    def init_embedding(self, init):
        nn.init.kaiming_normal_(
            self.ncf_user_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(
            self.ncf_item_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(
            self.gmf_user_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(
            self.gmf_item_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(
            self.hidden_layer1.weight, mode='fan_out', a=init)
        nn.init.constant_(self.hidden_layer1.bias, 0.0)

        nn.init.kaiming_normal_(
            self.output_layer.weight, mode='fan_out', a=init)

    def forward(self, users, items):

        ncf_u_latent = self.dropout(self.ncf_user_embedding(users))
        ncf_i_latent = self.dropout(self.ncf_item_embedding(items))
        ncf_ui_latent = torch.cat([ncf_u_latent, ncf_i_latent], 1)
        ncf_h = F.relu(self.hidden_layer1(ncf_ui_latent))
        gmf_u_latent = self.dropout(self.gmf_user_embedding(users))
        gmf_i_latent = self.dropout(self.gmf_item_embedding(items))
        gmf_h = gmf_u_latent * gmf_i_latent

        h = torch.cat([gmf_h, ncf_h], 1)
        preds = self.output_layer(h)

        preds = preds.squeeze(dim=-1)
        return preds

    def loss_function(self, preds, user_list, item_list, label_list):

        criterion = nn.MSELoss()
        loss = criterion(preds, label_list)
        return loss
