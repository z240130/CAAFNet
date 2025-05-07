import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torchvision import transforms
from PIL import Image

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        #return dice,hd95
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        #return 1,0
        return 1
    else:
        return 0

## save images ###
import numpy as np
from PIL import Image
import os

import os
import numpy as np
from PIL import Image
import cv2  # 添加 OpenCV 进行图像融合


def reassign_small_regions(label_img, area_threshold=50):
    """
    对每个类别进行连通域分析，对小区域用邻域多数投票重标。

    参数：
    - label_img: 形状为 (224, 244) 的标签图像，类别取值 0～9
    - area_threshold: 面积小于该值的连通区域将被重赋值

    返回：
    - processed: 处理后的标签图像
    """
    processed = label_img.copy()

    # 针对每个类别分别处理
    for cls in np.unique(label_img):
        # 创建当前类别的二值图像
        mask = (label_img == cls).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 遍历所有连通区域（跳过背景，i=0）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < area_threshold:
                # 获取当前连通区域的掩码
                region_mask = (labels == i)
                # 膨胀该区域以获得边缘邻域
                dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
                border_mask = (dilated.astype(bool)) & (~region_mask)
                # 取出邻域中的标签
                neighbor_labels = label_img[border_mask]
                if neighbor_labels.size > 0:
                    # 多数投票：选取出现次数最多的标签
                    majority_label = np.bincount(neighbor_labels).argmax()
                    # 重新赋值
                    processed[region_mask] = majority_label
    return processed

import os
import numpy as np
from PIL import Image

def Save_image_per_class(img, seg, ano, path, person, epoch):
    """
    img:    numpy array, shape [batch, C, H, W]
    seg:    numpy array, shape [batch, num_classes, H, W], 未 softmax 的 logits 或 softmax 后的概率
    ano:    numpy array, shape [batch, H, W], 真值标签类别索引
    path:   用于文件名标识（通常是样本索引）
    person: 用于文件名标识的次级索引
    epoch:  当前 epoch，用于分类保存目录
    """

    # 定义你原来的颜色映射
    color_map = {
        1: [0, 0, 255],     # 蓝色
        2: [0, 255, 0],     # 绿色
        3: [255, 0, 0],     # 红色
        4: [0, 255, 255],   # 青色
        5: [255, 0, 255],   # 紫色
        6: [255, 255, 0],   # 黄色
        7: [135, 206, 250], # 浅蓝色
        8: [255, 255, 255], # 白色
    }

    # 创建基础输出目录
    base_dir = f"result_unet/Image_9/Inputs_{epoch}"
    os.makedirs(base_dir, exist_ok=True)

    # 取 batch 的第一张
    img_np = img[0]                     # (C, H, W)
    img_vis = img_np.transpose(1, 2, 0) # -> (H, W, C)
    # 归一化到 0–255 并转 uint8
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
    img_vis = np.uint8(img_vis * 255)
    # 如果是单通道，将其扩展为 3 通道
    if img_vis.ndim == 2:
        img_vis = np.stack([img_vis] * 3, axis=-1)
    elif img_vis.ndim == 3 and img_vis.shape[2] == 1:
        img_vis = np.concatenate([img_vis] * 3, axis=2)

    # 取预测类别索引
    pred_cls = np.argmax(seg[0], axis=0)  # (H, W)
    # 真值类别
    true_cls = ano[0]                     # (H, W)

    # 对每个类别分别生成一张图
    for cls, color in color_map.items():
        # 构造当前类别的 mask
        mask = (pred_cls == cls)
        if not mask.any():
            # 该图中未出现此类别，就跳过
            continue

        # 在原图上覆盖当前颜色
        overlay = img_vis.copy()
        overlay[mask] = color

        # 保存文件
        save_path = os.path.join(
            base_dir,
            f"{path}_person{person}_class{cls}.png"
        )
        #Image.fromarray(overlay).save(save_path, quality=95)
    Image.fromarray(img_vis).save(f"result_unet/Image_9/1.png", quality=95)
    # （可选）如果还要保存真值同样的 8 张图，可仿照上面逻辑，把 pred_cls 换成 true_cls，
    # 并存到另一个子目录 True_{epoch} 中，使用 color_map 同样的颜色。

def Save_image(img, seg, ano, path,person,epoch):
    seg = np.argmax(seg, axis=1)  # (batch_size, H, W) 获取类别索引
    img = img[0]  # 取 batch 中的第一张 (C, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    seg = seg[0]  # 取 batch 中的第一张 (H, W)
    img = (img - img.min()) / (img.max() - img.min())
    #ano = reassign_small_regions(ano)
    # 颜色映射
    color_map = {
        1: [0, 0, 255],    # 蓝色
        2: [0, 255, 0],    # 绿色
        3: [255, 0, 0],    # 红色
        4: [0, 255, 255],  # 青色
        5: [255, 0, 255],  # 紫色
        6: [255, 255, 0],  # 黄色
        7: [135, 206, 250], # 浅蓝色
        8: [255, 255, 255]  # 白色
    }

    # 创建可视化图
    dst1 = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # 预测分割可视化
    #dst2 = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # 真实分割可视化
    for cls, color in color_map.items():
        dst1[seg == cls] = color
        #dst2[ano == cls] = color
    # 归一化原始图像
    img = np.uint8(img * 255.0)
    if img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)  # (H, W, 1) -> (H, W, 3)

    # 创建掩膜：仅在分割区域内覆盖颜色
    mask = np.any(dst1 != [0, 0, 0], axis=-1)  # True 表示该像素属于分割区域
    #mask1 =  np.any(dst2 != [0, 0, 0], axis=-1)
    overlay = img.copy()  # 复制原图
    ture_label = img.copy()
    overlay[mask] = dst1[mask]  # 仅在预测区域覆盖分割颜色
    #ture_label[mask1]=dst2[mask1]
    # 转换为 PIL 格式
    overlay = Image.fromarray(overlay)
    ture_label = Image.fromarray(ture_label)
    # 确保目录存在
    os.makedirs(f"result_unet/Image_9/Inputs_{epoch}", exist_ok=True)
    #os.makedirs("result_unet/Image/Ano0", exist_ok=True)
    # 保存
    overlay.save("result_unet/Image_9/Inputs_{}/{}_person{}.png".format(epoch,path, person), quality=95)
    #ture_label.save("result_unet/Image/Ano0/{}_person{}.png".format(path, person), quality=100)




def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,path=0,epoch=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            label_slice =label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
                label1 =zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=3)# previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(slice).unsqueeze(0).float().cuda()
            # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)  # forward
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                inputs = input.cpu().numpy()
                outputs1 = outputs.cpu().numpy()
                #if ind == 66 and path ==1:
                    #Save_image_per_class(inputs, outputs1, label1,ind + 1,path,epoch)
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list