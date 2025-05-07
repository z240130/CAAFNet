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
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
from datasets.dataloader import get_loader, test_dataset
import deeplake
from datasets.dataset_synapse import Synapse_dataset
from utils import DiceLoss, calculate_metric_percase
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
from dsc import DiceScoreCoefficient
from utils_MoNu import WeightedDiceBCE


import torch
import numpy as np
from PIL import Image
from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch, target_size=(224, 224)):
    new_batch = []
    for sample in batch:
        # 假设原始 sample 至少包含 'images' 和 'masks'
        image = sample['images']
        mask = sample['masks']

        # 将 mask 转换为 0 和 1
        if mask.dtype == bool:
            # 如果是布尔类型，直接转换即可 (True 变为 1, False 变为 0)
            mask = mask.astype(np.uint8)
        else:
            # 如果 mask 中的数值非布尔（如 0 与 255），将所有非零值归一为 1
            mask = mask.astype(np.uint8)
            mask = (mask > 0).astype(np.uint8)

        # 移除 mask 可能的单通道维度：形状 (H, W, 1) --> (H, W)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        # 将 numpy 数组转换为 PIL Image 并调整尺寸
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask, mode="L")

        image_resized = image_pil.resize(target_size)
        # 对 mask 使用最近邻插值，保持二值信息
        mask_resized = mask_pil.resize(target_size, resample=Image.NEAREST)

        # 转换为 PyTorch 张量
        new_sample = {}
        new_sample['image'] = torch.tensor(np.array(image_resized), dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        new_sample['label'] = torch.tensor(np.array(mask_resized), dtype=torch.long)

        new_batch.append(new_sample)

    return default_collate(new_batch)



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
def trainer_glas(args, model, snapshot_path):
    # 日志配置
    logging.basicConfig(filename=snapshot_path + "/log_glas.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    base_lr = args.base_lr
    num_classes = args.num_classes



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    ds = deeplake.load("hub://activeloop/glas-train", read_only=True)
    ds_test = deeplake.load("hub://activeloop/glas-test", read_only=True)
    dataloader = ds.pytorch(num_workers=0, batch_size=12, shuffle=False,collate_fn=custom_collate_fn)
    dataloader_test= ds_test.pytorch(num_workers=0, batch_size=1, shuffle=False,collate_fn=custom_collate_fn)
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

    best = 0.0
    loss_list = []
    dice_list = []
    for epoch_num in range(max_epoch):
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch_num}/{max_epoch}', unit='img', leave=True) as pbar:
            for batch_sampler in dataloader:

                image_batch, label_batch = batch_sampler['image'], batch_sampler['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()


                outputs = model(image_batch)
                out_loss = criterion(torch.sigmoid(outputs), label_batch.float())

                # 检查损失是否有效

                # 反向传播和优化
                optimizer.zero_grad()
                out_loss.backward()




                optimizer.step()



                pbar.update(1)
                pbar.set_postfix(loss=out_loss.item())
                iter_num += 1


                # 验证
            if epoch_num % 1 == 0:


                    logging.info("Validation ===>")

                    val = inference(model, test_save_path=None, val=dataloader_test,criterion=criterion,epoch=epoch_num)


                    logging.info('epoch: {}, dice: {}'.format(epoch_num, val))
                    loss_list.append(out_loss.item())
                    dice_list.append(val.detach().cpu().numpy().item())




                    if val > best:
                        #print('##################### Dice score improved from {} to {}'.format(best, meandice))
                        logging.info('##################### Dice score improved from {} to {}'.format(best, val))
                        best = val

    df = pd.DataFrame({
        'epoch': list(range(max_epoch)),
        'loss': loss_list,
        'dice': dice_list
    })
    df.to_csv(snapshot_path + '/uc_glas.csv', index=False)

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
    #seg = seg[0, 0, :, :]
    img = img[0]
    img_vis = img.transpose(1, 2, 0).astype(np.uint8)
    img_pil = Image.fromarray(img_vis)
    seg = ano[ 0, :, :]
    # 使用阈值0.5进行二值化：大于0.5为1（前景），否则为0（背景）
    seg_bw = (seg > 0.5).astype(np.uint8) * 255

    # 转换为 RGB 三通道（每个通道均相同）
    seg_rgb = np.stack([seg_bw] * 3, axis=-1)

    # 转换为 PIL Image（RGB模式）
    pred_rgb = Image.fromarray(seg_rgb, mode='RGB')

    # 创建保存目录（如果不存在）
    save_dir = f"result_unet/Image_GLAS/label"
    #save_dir = f"result_unet/Image_GLAS/Inputs_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"result_unet/Image_GLAS/input", exist_ok=True)
    img_pil.save(f"result_unet/Image_GLAS/input/{path}.png", quality=95)
    # 保存图像，质量设置为95
    pred_rgb.save(f"{save_dir}/{path}.png", quality=95)


def inference(model, test_save_path=None,val=None,criterion=None,epoch=None):
    model.eval()

    DSC = 0.0
    dice_sum=0.0
    with torch.no_grad():
        for i, val_sampled_batch in enumerate(val):
            image, gt = val_sampled_batch["image"], val_sampled_batch["label"]
            image, gt = image.cuda(), gt.cuda()
            outputs = model(image)  # 前向传播
            # 如果需要，可以转换为 numpy，保存图像等操作
            outputs1 = outputs.cpu().detach().numpy()
            image1 = image.cpu().numpy()
            gt1 = gt.cpu().numpy()
            Save_image(image1, outputs1, gt1, i + 1, epoch)

            # 计算 Dice 分数，注意此处不应触发梯度计算
            train_dice = criterion._show_dice(torch.sigmoid(outputs), gt.float())
            DSC += train_dice
    return DSC / len(val)
