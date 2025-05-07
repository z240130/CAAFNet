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

from datasets.dataset_ACDC import ACDCdataset, RandomGenerator
from datasets.dataset_synapse import Synapse_dataset
from utils import DiceLoss, calculate_metric_percase
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
from dsc import DiceScoreCoefficient
def trainer_ACDC(args, model, snapshot_path):
    # 日志配置
    logging.basicConfig(filename=snapshot_path + "/log_ACDC.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 数据集
    # x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # y_transforms = transforms.ToTensor()
    db_train = ACDCdataset(args.root_path, args.list_dir, split="train", transform=
    transforms.Compose(
        [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))
    db_val = ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")

    testloader = DataLoader(db_val, batch_size=12, shuffle=False, num_workers=0, drop_last=True)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=12, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)

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
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num=0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_inference = 0.0
    loss_list = []
    dice_list = []
    for epoch_num in range(max_epoch):
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch_num}/{max_epoch}', unit='img', leave=True) as pbar:
            for sampled_batch in trainloader:
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

                # 检查输入是否有效
                if torch.isnan(image_batch).any() or torch.isinf(image_batch).any():
                    logging.warning("输入数据包含 NaN 或 Inf，跳过当前迭代")
                    continue

                # 前向传播

                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

                # 检查损失是否有效

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()


                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                # Adjust learning rate
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num += 1

                # writer.add_scalar('info/lr', lr_, iter_num)
                # writer.add_scalar('info/total_loss', loss, iter_num)
                # # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                # logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
            # 验证
            if epoch_num % 1 == 0:
                val = inference(model, test_save_path=None,val=testloader,db_test=db_val,epoch=epoch_num)
                logging.info(f"epoch:{epoch_num}, Validation result: {val}, Best: {best_inference}")
                loss_list.append(loss.item())
                dice_list.append(val.item())
                if val > best_inference:
                    best_inference = val

    df = pd.DataFrame({
        'epoch': list(range(max_epoch)),
        'loss': loss_list,
        'dice': dice_list
    })
    df.to_csv(snapshot_path + '/uc_acdc.csv', index=False)

    writer.close()
    return "Training Finished!"
#
from PIL import Image
def Save_image(img, seg, ano, path,epoch):
    seg = np.argmax(seg, axis=1)  # (batch_size, H, W) 获取类别索引
    img = img[0]  # 取 batch 中的第一张 (C, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    seg = seg[0]  # 取 batch 中的第一张 (H, W)
    #seg = np.transpose(seg, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())
    ano = ano[0][0]
    # 颜色映射
    color_map = {
        1: [0, 0, 255],    # 蓝色
        2: [0, 255, 0],    # 绿色
        3: [255, 0, 0],    # 红色
        4: [0, 255, 255],  # 青色
        # 5: [255, 0, 255],  # 紫色
        # 6: [255, 255, 0],  # 黄色
        # 7: [135, 206, 250], # 浅蓝色
        # 8: [255, 255, 255]  # 白色
    }

    # 创建可视化图
    dst1 = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # 预测分割可视化
    dst2 = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # 真实分割可视化
    for cls, color in color_map.items():
        dst1[seg == cls] = color
        dst2[ano == cls] = color
    # 归一化原始图像
    img = np.uint8(img * 255.0)
    if img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)  # (H, W, 1) -> (H, W, 3)

    # 创建掩膜：仅在分割区域内覆盖颜色
    mask = np.any(dst1 != [0, 0, 0], axis=-1)  # True 表示该像素属于分割区域
    mask1 =  np.any(dst2 != [0, 0, 0], axis=-1)
    overlay = img.copy()  # 复制原图
    ture_label = img.copy()
    overlay[mask] = dst1[mask]  # 仅在预测区域覆盖分割颜色
    ture_label[mask1]=dst2[mask1]
    # 转换为 PIL 格式
    overlay = Image.fromarray(overlay)
    ture_label = Image.fromarray(ture_label)
    # 确保目录存在
    os.makedirs(f"result_unet/Image_ACDC/Inputs_{epoch}", exist_ok=True)
    os.makedirs("result_unet/Image_ACDC/label/", exist_ok=True)
    # 保存
    overlay.save("result_unet/Image_ACDC/Inputs_{}/{}.png".format(epoch,path), quality=95)
    ture_label.save("result_unet/Image_ACDC/label/{}.png".format(path), quality=100)

def inference(model, test_save_path=None,val=None,db_test=None,epoch=None):
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    device="cuda:0"
    model.eval()
    predict = []
    answer = []
    metric=0.0
    for i, val_sampled_batch in enumerate(val):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
            torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.to(device).unsqueeze(1), val_label_batch.to(
            device).unsqueeze(1)
        val_outputs = model(val_image_batch)
        val_image_batch = val_image_batch.cpu().detach().numpy()
        label = val_label_batch.cpu().detach().numpy()
        #val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        val_outputs = torch.softmax(val_outputs, dim=1)
        prediction = val_outputs.cpu().detach().numpy()
        val_label_batch = val_label_batch.squeeze(0).cpu().detach().numpy()

        #Save_image(val_image_batch, prediction, label, i + 1,epoch)


        for j in range(12):
             predict.append(prediction[j])
             answer.append(val_label_batch[j])
    # predict.append(prediction)
    # answer.append(val_label_batch)
    #performance = dc_sum /  len(val)
    #logging.info('Testing performance in val model: mean_dice : %f, ' % (performance))
    dsc = DiceScoreCoefficient(n_classes=4)(predict, answer)

    print("Dice")



    # best_dsc = np.zeros_like(dsc)  # 初始化最优记录

    # 更新最优记录
    # best_dsc = np.maximum(best_dsc, dsc)

    # print("Dice")
    # for i in range(len(dsc)):
    #     print("class %d  = %f  (best = %f)" % (i, dsc[i], best_dsc[i]))
    print("mDice     = %f" % (np.mean(dsc)))


    # print("val avg_dsc: %f" % (performance))
    return np.mean(dsc)
    #return 0