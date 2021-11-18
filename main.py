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
    parser.add_argument('--version', type=str, default='0.3')
    parser.add_argument('--note', type=str, default='Finish modeling flow.')

    # Model settings
    parser.add_argument('--bin_sizes', type=list, default=[2, 3, 6])
    parser.add_argument('--pooling_dim', type=int, default=512)
    parser.add_argument('--enable_spp', type=bool, default=False)
    parser.add_argument('--recalculate_mean_std', type=bool, default=False)

    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--gamma_steplr', type=float, default=np.sqrt(0.1))
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)

    # Settings need to be tuned
    parser.add_argument('--backbone', type=str, default='resnet50')  # Num of cross validation folds
    parser.add_argument('--data', default='assd')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--manual_lr', type=bool, default=False)  # Will override other lr
    parser.add_argument('--manual_lr_epoch', type=int, default=5)
    parser.add_argument('--smooth_label', type=float, default=0.3)
    parser.add_argument('--alpha_loss', type=float, default=0.5)

    # Augmentation
    parser.add_argument('--enable_hvflip', type=float, default=0.5)  # enable horizontal and vertical flipping
    parser.add_argument('--enable_wgn', type=float, default=0.5)  # apply white Gaussian noise

    opt = parser.parse_args()
    opt.log = '_result/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'
    opt.device = torch.device('cuda')

    # Model settings
    if opt.backbone == 'resnet50':
        opt.out_dim_resnet = 512
    opt.seg_criterion = nn.CrossEntropyLoss().to(opt.device)
    opt.cls_criterion = nn.BCEWithLogitsLoss().to(opt.device)

    # Print hyperparameters and settings
    print('\n[Info] Model settings:\n')
    for k, v in vars(opt).items():
        print('         %s: %s' % (k, v))

    with open(opt.log, 'a') as f:
        # Save hyperparameters
        for k, v in vars(opt).items():
            f.write('%s: %s\n' % (k, v))

    """
    Start modeling
    """
    # Import data
    data_getter = getter_dataloader(opt)
    (opt.num_label, opt.h, opt.w) = get_data_detail(opt.data)

    with open(opt.log, 'a') as f:
        f.write('\nEpoch, Time, loss_tr, miou_tr, acc_tr, loss_val, miou_val, acc_val\n')

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
    best_loss = 1e9
    best_miou = 0
    best_acc = 0

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

        loss_train, miou_train, acc_train = train_epoch(model, trainloader, opt, optimizer)

        if not opt.manual_lr:
            scheduler.step()

        end = time.time()

        print('\n- (Training) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, accuracy:{acc: 8.4f}, elapse:{elapse:3.4f} min'
              .format(loss=loss_train, miou=miou_train, acc=acc_train, elapse=(time.time() - start) / 60))

        """ Validating """
        with torch.no_grad():
            loss_val, miou_val, acc_val = test_epoch(model, valloader, opt)

        print('\n- (Validating) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, accuracy:{acc: 8.4f}'
              .format(loss=loss_val, miou=miou_val, acc=acc_val))

        """ Logging """
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {miou_train: 8.4f}, {acc_train: 8.4f}, '
                    '{loss_val: 8.5f}, {miou_val: 8.4f}, {acc_val: 8.4f}\n'
                    .format(epoch=epoch, time=(end - start) / 60, loss_train=loss_train, miou_train=miou_train,
                            acc_train=acc_train, loss_val=loss_val, miou_val=miou_val, acc_val=acc_val), )

        """ Early stopping """
        if epoch > opt.manual_lr_epoch:
            if best_miou < miou_val or (best_miou == miou_val) & (best_loss >= loss_val):
                best_loss = loss_val
                best_miou = miou_val
                best_acc = acc_val

                patience = 0
                print("\n- New best performance logged.")
            else:
                patience += 1
                print("\n- Early stopping patience counter {} of {}".format(patience, opt.es_patience))

                if patience == opt.es_patience:
                    print("\n[Info] Stop training")
                    break
        else:
            print("\n- Warming up learning rate.")

        print("\n[Info] Training stopped with best loss: {loss: 8.5f}, best miou: {miou: 8.4f} "
              "and best accuracy: {acc: 8.4f}\n"
              .format(loss=best_loss, miou=best_miou, acc=best_acc), )

        with open(opt.log, 'a') as f:
            f.write("\n[Info] Training stopped with best loss: {loss: 8.5f}, best miou: {miou: 8.4f} "
                    "and best accuracy: {acc: 8.4f}"
                    .format(loss=best_loss, miou=best_miou, acc=best_acc), )


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
