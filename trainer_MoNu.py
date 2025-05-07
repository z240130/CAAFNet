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
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
from datasets.dataloader import get_loader, test_dataset

from datasets.dataset_synapse import Synapse_dataset
from utils import DiceLoss, calculate_metric_percase
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
from dsc import DiceScoreCoefficient
from utils_MoNu import WeightedDiceBCE


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
def trainer_MoNu(args, model, snapshot_path):
    # 日志配置
    logging.basicConfig(filename=snapshot_path + "/log_MoNu.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    base_lr = args.base_lr
    num_classes = args.num_classes



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    train_tf = transforms.Compose([RandomGenerator(output_size=[224, 224])])
    val_tf = ValGenerator(output_size=[224, 224])
    train_dataset = ImageToImage2D( './datasets/'+ 'MoNuSeg'+ '/Train_Folder/', train_tf, image_size=224)
    val_dataset = ImageToImage2D('./datasets/'+ 'MoNuSeg'+ '/Val_Folder/', val_tf, image_size=224)
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=0,
                            pin_memory=True)

    print("#" * 20, "Start Training", "#" * 20)


    # 模型配置
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device="cuda:0")
    model.train()

    # 损失函数和优化器


    optimizer = torch.optim.AdamW(params = model.parameters(), lr=1e-4, weight_decay=1e-4)
    writer = SummaryWriter(snapshot_path + '/log')
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    iter_num=0
    max_epoch = args.max_epochs
    print(max_epoch)
    best = 0.0

    # 初始化 loss 和 dice 的记录列表
    loss_list = []
    dice_list = []
    for epoch_num in range(max_epoch):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch_num}/{max_epoch}', unit='img', leave=True) as pbar:
            for (sampled_batch, names) in train_loader:

                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()


                outputs = model(image_batch)
                out_loss = criterion(torch.sigmoid(outputs), label_batch.float())
                # loss_ce = F.binary_cross_entropy_with_logits(outputs.squeeze(), label_batch.float(),reduce='none')
                # loss_dice = binary_dice_loss(outputs, label_batch)
                # loss = 0.4 * loss_ce + 0.6 * loss_dice

                # 检查损失是否有效

                # 反向传播和优化
                optimizer.zero_grad()
                out_loss.backward()

                # 梯度裁剪
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()



                pbar.update(1)
                pbar.set_postfix(loss=out_loss.item())
                iter_num += 1


                # 验证
            if epoch_num % 1 == 0:


                    logging.info("Validation ===>")

                    val = inference(model, test_save_path=None, val=val_loader,criterion=criterion,epoch=epoch_num)


                    logging.info('epoch: {}, dice: {}'.format(epoch_num, val))
                    # 记录 loss 和 dice
                    loss_list.append(out_loss.item())
                    dice_list.append(val.detach().cpu().numpy().item())




                    if val > best:
                        #print('##################### Dice score improved from {} to {}'.format(best, meandice))
                        logging.info('##################### Dice score improved from {} to {}'.format(best, val))
                        best = val

        # 保存 loss 和 dice 的日志为 CSV
    df = pd.DataFrame({
        'epoch': list(range(max_epoch)),
        'loss': loss_list,
        'dice': dice_list
    })
    df.to_csv(snapshot_path + '/UC_net_metrics_log.csv', index=False)

    writer.close()
    return "Training Finished!"
#
from PIL import Image
import os
import numpy as np
from PIL import Image

import numpy as np
import os
from PIL import Image

import numpy as np
import os
from PIL import Image


def Save_image(img, seg, ano, path, epoch):
    # 取 batch 中的第一张图像 (1, H, W)
    # seg 的 shape 应为 [batch, 1, height, width]
    seg = seg[0, 0, :, :]

    # 使用阈值0.5进行二值化：大于0.5为1（前景），否则为0（背景）
    seg_bw = (seg > 0.5).astype(np.uint8) * 255

    # 转换为 RGB 三通道（每个通道均相同）
    seg_rgb = np.stack([seg_bw] * 3, axis=-1)

    # 转换为 PIL Image（RGB模式）
    pred_rgb = Image.fromarray(seg_rgb, mode='RGB')

    # 创建保存目录（如果不存在）
    save_dir = f"result_unet/Image_MONU/Inputs_{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存图像，质量设置为95
    pred_rgb.save(f"{save_dir}/{path}.png", quality=95)


def inference(model, test_save_path=None,val=None,criterion=None,epoch=None):


    DSC = 0.0
    dice_sum=0.0
    for i, (val_sampled_batch,name) in enumerate(val):
        image, gt = val_sampled_batch["image"], val_sampled_batch["label"]
        #gt = gt.cpu().numpy()
        image,gt = image.cuda(),gt.cuda()
        outputs = model(image)  # forward
        outputs1 = outputs.cpu().detach().numpy()
        image1 = image.cpu().numpy()
        gt1 = gt.cpu().numpy()
        #Save_image(image1, outputs1, gt1, i + 1,  epoch)
        train_dice = criterion._show_dice(torch.sigmoid(outputs), gt.float())


        DSC = DSC + train_dice

    return DSC/len(val)
