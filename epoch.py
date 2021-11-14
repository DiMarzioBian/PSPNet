import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm


def train_epoch(model, data, opt, optimizer):
    """
    Flow for each epoch
    """
    num_data = data.dataset.length
    num_pred_correct_epoch = 0
    loss_epoch = 0
    model.train()
    for batch in tqdm(data, desc='- (Training)   ', leave=False):
        wave, y_gt = map(lambda x: x.to(opt.device), batch)

        """ training """
        loss_batch, num_pred_correct_batch = model.loss(wave, y_gt)
        loss = loss_batch / batch.__len__()
        loss.backward()

        num_pred_correct_epoch += num_pred_correct_batch
        loss_epoch += loss_batch

    return loss_epoch / num_data, num_pred_correct_epoch / num_data


def test_epoch(model, data, gt_voting, opt, dataset):
    """
    Give prediction on test set
    """
    num_data = data.dataset.length
    num_pred_correct_epoch = 0
    loss_epoch = 0

    model.eval()
    for batch in tqdm(data, desc='- (Testing)   ', leave=False):
        wave, y_gt_batch = map(lambda x: x.to(opt.device), batch)

        loss_batch, num_pred_correct_batch, y_pred_batch = model.predict(wave, y_gt_batch)
        num_pred_correct_epoch += num_pred_correct_batch
        loss_epoch += loss_batch

    return loss_epoch / num_data, num_pred_correct_epoch / num_data
