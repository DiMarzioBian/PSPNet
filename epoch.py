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
    acc_epoch = 0

    model.train()
    for batch in tqdm(data, desc='- (Training)   ', leave=False):
        wave, y_gt = map(lambda x: x.to(opt.device), batch)

        """ training """
        y_pred = model(wave, y_gt)
        loss = get_loss(y_pred, y_gt)
        miou_batch, acc_batch = get_metric(y_pred, y_gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_epoch += loss
        miou_epoch += miou_batch
        acc_epoch += acc_batch

    return loss_epoch / num_data, miou_epoch / num_data, acc_epoch / num_data


def test_epoch(model, data, gt_voting, opt, dataset):
    """
    Give prediction on test set
    """
    num_data = data.dataset.length
    loss_epoch = 0
    miou_epoch = 0
    acc_epoch = 0

    model.eval()
    for batch in tqdm(data, desc='- (Testing)   ', leave=False):
        wave, y_gt = map(lambda x: x.to(opt.device), batch)

        """ training """
        y_pred = model(wave, y_gt)
        loss = get_loss(y_pred, y_gt)
        miou_batch, acc_batch = get_metric(y_pred, y_gt)

        loss_epoch += loss
        miou_epoch += miou_batch
        acc_epoch += acc_batch

    return loss_epoch / num_data, miou_epoch / num_data, acc_epoch / num_data
