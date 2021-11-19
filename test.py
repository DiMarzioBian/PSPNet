import argparse
import numpy as np

from PIL import Image
import torch
from torchvision import transforms as T
from model.pspnet import PSPNet


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
    parser.add_argument('--bin_sizes', type=list, default=[2, 3, 6])
    parser.add_argument('--enable_aux', type=bool, default=True)
    parser.add_argument('--alpha_loss', type=float, default=0.4)

    parser.add_argument('--name_mode_dict', default='x')

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

    # Load model
    model = PSPNet(opt)
    model.load_state_dict('_result/model/'+opt.name_mode_dict)
    model = model.to(opt.device)
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.44619015, 0.44376444, 0.40185362], std=[0.20309216, 0.19916435, 0.209552])
    ])
    if opt.shrink_image:
        pre_transform = T.Compose([
            T.Resize(opt.shrink_image, T.InterpolationMode.NEAREST)
        ])

    # Load images

    index = '000'
    path_gt = '_data/assd/label_images_semantic/' + index + '.png'

    image = train_transform(np.array(Image.open(index + '.jpg')) / 255)
    gt = torch.Tensor(np.array(Image.open(path_gt)))

    if opt.shrink_image:
        image = pre_transform(image)
        gt = pre_transform(gt)

    # Predicting labels
    y_score, y_score_aux = model(image)

    # Plot


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
