from model.pspnet import *
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model.pspnet import PSPNet
from datasets import getter_dataloader, get_data_detail
from epoch import train_epoch, test_epoch
from utils import set_optimizer_lr, update_optimizer_lr


def main():
    """
    Preparation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--shrink_image', type=list, default=(400, 600))
    parser.add_argument('--backbone', type=str, default='resnet18')  # Num of cross validation folds
    parser.add_argument('--data', default='assd')

    opt = parser.parse_args()
    opt.device = torch.device('cuda')

    # Model settings
    if opt.backbone == 'resnet18':
        opt.out_dim_resnet = 512
        opt.out_dim_resnet_auxiliary = 256
        opt.out_dim_pooling = 512
    elif opt.backbone == 'resnet34':
        opt.out_dim_resnet = 512
        opt.out_dim_resnet_auxiliary = 256
        opt.out_dim_pooling = 512
    elif opt.backbone == 'resnet50':
        opt.out_dim_resnet = 2048
        opt.out_dim_resnet_auxiliary = 1024
        opt.out_dim_pooling = 2048


    data_getter = getter_dataloader(opt)
    (opt.num_label, opt.h, opt.w) = get_data_detail(opt.data)

    with open(opt.log, 'a') as f:
        f.write('\nEpoch, Time, loss_tr, loss_aux_tr, miou_tr, acc_tr, loss_val, miou_val, acc_val\n')

    # Load model
    model = PSPNet(opt)
    model = model.to(opt.device)
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9,
                          weight_decay=opt.l2_reg, nesterov=True)
    if not opt.manual_lr:
        scheduler = optim.lr_scheduler.StepLR(optimizer, int(opt.lr_patience), gamma=opt.gamma_steplr)

    # Load data
    print('\n[Info] Loading data...')
    trainloader, valloader, val_gt_voting = data_getter.get()

    # Define logging variants
    loss_best = 1e9
    aux_best = 1e9
    miou_best = 0
    pa_best = 0

    for epoch in range(1, opt.epoch + 1):
        print('\n[ Epoch {epoch}]'.format(epoch=epoch))

        # """ Training """
        start = time.time()

        if opt.manual_lr and epoch <= opt.manual_lr_epoch:
            set_optimizer_lr(optimizer, 1e-5)
        elif opt.manual_lr and epoch == (opt.manual_lr_epoch + 1):
            set_optimizer_lr(optimizer, 5e-3)
        elif opt.manual_lr and (epoch - opt.manual_lr_epoch) % 30 == 0:
            update_optimizer_lr(optimizer)

        loss_train, loss_aux_train, miou_train, pa_train = train_epoch(model, trainloader, opt, optimizer)

        if not opt.manual_lr:
            scheduler.step()

        end = time.time()

        """ Validating """
        with torch.no_grad():
            loss_val, miou_val, pa_val = test_epoch(model, valloader, opt)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
