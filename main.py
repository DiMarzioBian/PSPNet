from model.pspnet import *
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model.pspnet import PSPNet
from datasets import getter_dataloader, get_num_label
from epoch import train_epoch, test_epoch
from utils import set_optimizer_lr, update_optimizer_lr


def main():
    """
    Preparation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='0.3')
    parser.add_argument('--note', type=str, default='Finish modeling flow.')

    parser.add_argument('--smooth_label', type=float, default=0.3)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--gamma_steplr', type=float, default=np.sqrt(0.1))
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--is_distributed', type=bool, default=False)

    # Settings need to be tuned
    parser.add_argument('--cv', type=int, default=1)  # Num of cross validation folds
    parser.add_argument('--data', default='assd')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--manual_lr', type=bool, default=False)  # Will override other lr
    parser.add_argument('--manual_lr_epoch', type=int, default=5)

    opt = parser.parse_args()
    opt.log = '_result/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'
    opt.device = torch.device('cuda')

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
    opt.num_label = get_num_label(opt.data)

    cv_acc = np.zeros(opt.cv)
    cv_miou = np.zeros(opt.cv)
    cv_loss = np.zeros(opt.cv)

    """ Iterate 10 folds """
    for fold in range(opt.cv):
        print("\n------------------------ Start fold:{fold} ------------------------\n".format(fold=fold))
        with open(opt.log, 'a') as f:
            f.write("\n------------------------ Start fold:{fold} ------------------------\n".format(fold=fold), )
            f.write('\nEpoch, Time, loss_tr, miou_tr, acc_tr, loss_val, miou_val, acc_val\n')

        # Load Music model
        model = PSPNet(opt).to(opt.device)
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9,
                              weight_decay=opt.l2_reg, nesterov=True)
        if not opt.manual_lr:
            scheduler = optim.lr_scheduler.StepLR(optimizer, int(opt.lr_patience), gamma=opt.gamma_steplr)

        # Load data
        print('\n[Info] Loading data...')
        trainloader, valloader, val_gt_voting = data_getter.get(fold)

        # Define logging variants
        best_loss = 1e9
        best_miou = 0
        best_acc = 0

        for epoch in range(1, opt.epoch + 1):
            print('\n[ Epoch {epoch}]'.format(epoch=epoch))

            # """ Training """
            start = time.time()

            trainloader.dataset.shuffle()

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
                loss_val, miou_val, acc_val = test_epoch(model, valloader, val_gt_voting, opt, dataset='val')

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

        """ Logging """
        cv_loss[fold] = best_loss
        cv_miou[fold] = best_miou
        cv_acc[fold] = best_acc

        print("\n[Info] Training stopped with best loss: {loss: 8.5f}, best miou: {miou: 8.4f} "
              "and best accuracy: {acc: 8.4f}\n"
              .format(loss=best_loss, miou=best_miou, acc=best_acc), )

        with open(opt.log, 'a') as f:
            f.write("\n[Info] Training stopped with best loss: {loss: 8.5f}, best miou: {miou: 8.4f} "
                    "and best accuracy: {acc: 8.4f}"
                    .format(loss=best_loss, miou=best_miou, acc=best_acc), )

            f.write("\n------------------------ Finished fold:{fold} ------------------------\n"
                    .format(fold=fold), )
        print("\n------------------------ Finished fold:{fold} ------------------------\n".format(fold=fold))

    """ Final logging """
    with open(opt.log, 'a') as f:
        f.write("\n[Info] Average cross validation loss: {loss: 8.5f}, best miou: {acc: 8.4f} "
                "and best accuracy: {acc: 8.4f}\n"
                .format(loss=np.mean(cv_loss), miou=np.mean(cv_miou), acc=np.mean(cv_acc)), )
    print('\n[Info] 10-fold average: loss: {loss: 8.5f}, average miou: {miou: 8.4f} and average accuracy: {acc: 8.4f}\n'
          .format(loss=np.mean(cv_loss), miou=np.mean(cv_miou), acc=np.mean(cv_acc)), )
    print('\n------------------------ Finished. ------------------------\n')


# def test(opt, model, data_getter):
#     print('\n[ Epoch testing ]')
#
#     with torch.no_grad():
#         loss_test, acc_test, acc_test_voting = test_epoch(model, valloader, val_gt_voting, opt, dataset='test')
#
#     print('\n- [Info] Test loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
#           .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )
#
#     with open(opt.log, 'a') as f:
#         f.write('\nTest loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
#                 .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
