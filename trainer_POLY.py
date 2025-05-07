import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from medpy.metric import dc
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import get_loader, test_dataset
from datasets.dataset_ACDC import ACDCdataset, RandomGenerator
from datasets.dataset_synapse import Synapse_dataset
from utils import DiceLoss, calculate_metric_percase
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
from dsc import DiceScoreCoefficient

def binary_dice_loss(inputs, targets, smooth=1):
    """
    计算二分类 Dice Loss
    :param inputs: 预测值，形状 [B, 1, H, W]，需要经过 sigmoid
    :param targets: 真实标签，形状 [B, H, W]，0/1 二值
    :param smooth: 平滑项，防止分母为 0
    :return: Dice Loss
    """
    inputs = torch.sigmoid(inputs)  # 对输出进行 Sigmoid 归一化
    inputs = inputs.view(-1)  # 展平成 1D
    targets = targets.view(-1)  # 展平成 1D

    intersection = (inputs * targets).sum()  # 计算交集
    dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  # Dice 系数
    dice_loss = 1 - dice_score  # Dice Loss

    return dice_loss
def trainer_POLY(args, model, snapshot_path):
    # 日志配置
    logging.basicConfig(filename=snapshot_path + "/log_poly.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 数据集
    # x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # y_transforms = transforms.ToTensor()
    #db_train = ACDCdataset(args.root_path, args.list_dir, split="train", transform=
    # transforms.Compose(
    #     [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    # print("The length of train set is: {}".format(len(db_train)))
    # db_val = ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    #
    # testloader = DataLoader(db_val, batch_size=12, shuffle=False, num_workers=0, drop_last=True)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                          worker_init_fn=worker_init_fn, drop_last=True)
    train_loader = get_loader("./dataset/TrainDataset/images/", "./dataset/TrainDataset/masks/", batchsize=12, trainsize=352,
                              augmentation=False)
    total_step = len(train_loader)
    #testloader = test_dataset("./dataset/TestDataset/images/", "./dataset/TestDataset/masks/", 352)
    print("#" * 20, "Start Training", "#" * 20)


    # 模型配置
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device="cuda:0")
    model.train()

    # 损失函数和优化器
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)  # 加入平滑项
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=1e-4, weight_decay=1e-4)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num=0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    best = 0.0
    size_rates = [0.75, 1, 1.25]
    loss_list = []
    dice_list = []
    for epoch_num in range(max_epoch):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch_num}/{max_epoch}', unit='img', leave=True) as pbar:
            for sampled_batch in train_loader:
                #for rate in size_rates:
                image_batch, label_batch = sampled_batch[0],sampled_batch[1]
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                # trainsize = int(round(352 * rate / 32) * 32)
                # if rate != 1:
                #     images = F.upsample(image_batch, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                #     gts = F.upsample(label_batch, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                #     gts =gts.squeeze()
                outputs = model(image_batch)

                loss_ce = F.binary_cross_entropy_with_logits(outputs.squeeze(), label_batch.squeeze(),reduce='none')
                loss_dice = binary_dice_loss(outputs, label_batch)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

                # 检查损失是否有效

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()



                # Adjust learning rate
                #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                iter_num += 1

                    # writer.add_scalar('info/lr', lr_, iter_num)
                    # writer.add_scalar('info/total_loss', loss, iter_num)
                    # # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                    # logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))

                # 验证
            if epoch_num % 1 == 0:
                    total_dice = 0
                    total_images = 0
                    logging.info("Validation ===>")
                    for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                        dataset_dice, n_images = inference(model,"./dataset/TestDataset/", dataset)
                        total_dice += (n_images * dataset_dice)
                        total_images += n_images
                        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch_num, dataset, dataset_dice))
                        #print(dataset, ': ', dataset_dice)

                    meandice = total_dice / total_images
                    loss_list.append(loss.item())
                    dice_list.append(meandice)
                    #print('Validation dice score: {}'.format(meandice))
                    logging.info('Validation dice score: {}'.format(meandice))
                    if meandice > best:
                        #print('##################### Dice score improved from {} to {}'.format(best, meandice))
                        logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
                        best = meandice


    df = pd.DataFrame({
        'epoch': list(range(max_epoch)),
        'loss': loss_list,
        'dice': dice_list
    })
    df.to_csv(snapshot_path + '/uc_poly.csv', index=False)

    writer.close()
    return "Training Finished!"
#

def inference(model, path=None,dataset=None):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        outputs = model(image)  # forward

        res = F.upsample(outputs, size=gt.shape, mode='bilinear',
                         align_corners=False)  # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze()  # apply sigmoid aggregation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # eval Dice
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1
