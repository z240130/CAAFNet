import pandas as pd
import matplotlib.pyplot as plt

# 文件路径和标签
files = ['./model_out/new_our_mono.csv', './model_out/updated_file.csv', './model_out/new_uc_mono.csv']
labels = ['CAAFNet(Ours)', 'U-Net', 'UCTransNet']
colors = ['blue', 'green', 'red']

# 初始化图表
plt.figure(figsize=(12, 5))

# ----------- 画 DICE 曲线 -----------
plt.subplot(1, 2, 1)
for file, label, color in zip(files, labels, colors):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['dice'], label=label, color=color)
plt.title('Dice Score vs Epoch on MoNuSeg')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True)

# ----------- 画 LOSS 曲线 -----------
plt.subplot(1, 2, 2)
for file, label, color in zip(files, labels, colors):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['loss'], label=label, color=color)
plt.title('Loss vs Epoch on MoNuSeg')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
