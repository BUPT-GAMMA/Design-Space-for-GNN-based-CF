import torch.nn as nn

from graphgym.register import register, register_loss

from graphgym.config import cfg


def loss_example(pred, true):
    if cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred

def loss_mae(pred, true):
    if cfg.model.loss_fun == 'mae':
        l1_loss = nn.L1Loss(reduction=cfg.model.size_average)
        loss = l1_loss(pred, true)
        return loss, pred


register_loss('smoothl1', loss_example)
register_loss('mae', loss_mae)
