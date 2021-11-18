import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from model.metrics import *


def train_epoch(model, data, opt, optimizer):
    """
    Flow for each epoch
    """
    num_data = data.dataset.length
    loss_epoch = 0
    miou_epoch = 0
    pa_epoch = 0

    model.train()
    for batch in tqdm(data, desc='- (Training)   ', leave=False):
        images, y_gt = map(lambda x: x.to(opt.device), batch)

        """ training """
        y_score, y_score_auxiliary = model(images)

        loss_batch = opt.seg_criterion(y_score, y_gt.squeeze(1).long())
        loss_auxiliary_batch = opt.seg_criterion(y_score_auxiliary, y_gt.squeeze(1).long())
        loss = loss_batch + opt.alpha_loss * loss_auxiliary_batch

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred = y_score.argmax(1)
        miou_batch, pa_batch = get_metrics(y_pred, y_gt, opt.num_label)

        loss_epoch += loss_batch * images.shape[0]
        miou_epoch += miou_batch * images.shape[0]
        pa_epoch += pa_batch * images.shape[0]

    return loss_epoch / num_data, miou_epoch / num_data, pa_epoch / num_data


def test_epoch(model, data, opt):
    """
    Give prediction on test set
    """
    num_data = data.dataset.length
    loss_epoch = 0
    miou_epoch = 0
    acc_epoch = 0

    model.eval()
    for batch in tqdm(data, desc='- (Testing)   ', leave=False):
        images, y_gt = map(lambda x: x.to(opt.device), batch)

        """ training """
        y_score, y_scored_auxiliary = model(images, y_gt)
        loss_batch = get_loss(y_score, y_gt)
        miou_batch, acc_batch = get_metric(y_score, y_gt)

        loss_epoch += loss
        miou_epoch += miou_batch
        acc_epoch += acc_batch

    return loss_epoch / num_data, miou_epoch / num_data, acc_epoch / num_data
