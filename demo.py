import argparse
import os
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from torchviz import make_dot
import torch
from torchvision import transforms as T
from model.pspnet import PSPNet
from datasets import get_data_detail


def main():
    """
    Preparation
    """
    parser = argparse.ArgumentParser()
    # Model settings
    parser.add_argument('--shrink_image', type=list, default=[400, 600])
    # Settings need to be tuned
    parser.add_argument('--backbone', type=str, default='resnet18')  # Num of cross validation folds
    parser.add_argument('--data', default='assd')
    parser.add_argument('--bin_sizes', type=list, default=[1, 2, 3, 6])
    parser.add_argument('--enable_aux', type=bool, default=True)
    parser.add_argument('--alpha_loss', type=float, default=0.2)
    parser.add_argument('--backbone_freeze', type=bool, default=True)
    parser.add_argument('--name_mode_dict', default='demo.pkl')

    opt = parser.parse_args()
    opt.device = torch.device('cuda')
    opt.seg_criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    # Model settings
    (opt.num_label, opt.h, opt.w) = get_data_detail(opt.data)
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
    else:
        raise RuntimeError('\n[warning] Wrong demo settings.\n')

    # Load model
    model = PSPNet(opt)
    with open('_result/model/'+opt.name_mode_dict, 'rb') as f:
        dict_demo = pickle.load(f)
    # loaded_state = torch.load('_result/model/'+opt.name_mode_dict, map_location='cuda:0')
    model.load_state_dict(dict_demo)
    model = model.to(opt.device)

    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.44619015, 0.44376444, 0.40185362], std=[0.20309216, 0.19916435, 0.209552])
    ])
    pre_transform = T.Compose([
        T.Resize(opt.shrink_image, T.InterpolationMode.NEAREST)
    ])

    # Load images
    index = '000'
    path_img = '_data/assd/original_images/' + index + '.jpg'
    path_gt = '_data/assd/label_images_semantic/' + index + '.png'

    image = train_transform(np.array(Image.open(path_img)) / 255).to(opt.device)
    gt = torch.Tensor(np.array(Image.open(path_gt))).unsqueeze(0).to(opt.device)

    image = pre_transform(image).float().unsqueeze(0)
    y_gt = pre_transform(gt).float()

    # Predicting labels
    with torch.no_grad():
        y_score, _ = model(image)

    # Plot
    y_pred = y_score.argmax(-1)
    loss_batch = opt.seg_criterion(y_score, y_gt.squeeze(1).long())


def display(display_list):
    plt.figure(figsize=(12, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i].cpu().numpy())
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
