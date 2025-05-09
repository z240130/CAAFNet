# CAAFNet
Code for paper "Enhancing Medical Image Segmentation with CAAFNet: A Dual-Branch CNN-Transformer Fusion Architecture". 

## 1. Environment

- Please prepare an environment with Ubuntu 20.04, with Python 3.6.13, PyTorch 1.8.0, and CUDA 11.1.1.
## 2. Prepare data
The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (Synapse/BTCV: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd and ACDC: https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4).
## 3. Train/Test

- Train

```bash
python train.py --dataset Synapse --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.001 --batch_size 12
```

- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.001 --img_size 224 --batch_size 12
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
## Citations
```bibtex
@article{li2025caafnet,
  title={Enhancing Medical Image Segmentation with CAAFNet: A Dual-Branch CNN-Transformer Fusion Architecture},
  author={Li, Xiaoqing and Huo, Hua and Zhang,Chen},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}

```