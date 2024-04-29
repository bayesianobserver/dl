"""
Simple feedforward NN
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ffmodel(nn.Module):
    """ Feedforward Neural Network """

    @staticmethod
    def get_default_config():
        C = CN()
        # the following  options need to be provided externally
        C.input_size = None
        C.use_dropout = None
        C.p_drop = None
        return C

    def __init__(self, config):
        super().__init__()
        self.use_dropout = config.use_dropout
        self.linear1 = nn.Linear(config.input_size, config.input_size * 2)
        self.linear2 = nn.Linear(config.input_size * 2, config.input_size * 3)
        self.linear3 = nn.Linear(config.input_size * 3, config.input_size * 2)
        self.linear4 = nn.Linear(config.input_size * 2, 1)
        self.dropout = nn.Dropout(config.p_drop)
        self.act = nn.ReLU()
        
        #self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, targets=None):
        x = self.linear1(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear2(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear3(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear4(x)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            x = x.squeeze()
            loss_function = nn.MSELoss()
            loss = loss_function(x, targets)
        return x, loss



    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    
    