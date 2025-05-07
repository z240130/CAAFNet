import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_conv_diagram():
    fig, ax = plt.subplots(figsize=(6, 4))

    # 设置输入矩阵参数（5x5网格）
    input_origin = (0, 2)  # 左上角起始坐标
    square_size = 1
    rows, cols = 5, 5
    for i in range(rows):
        for j in range(cols):
            rect = patches.Rectangle((input_origin[0] + j * square_size, input_origin[1] - i * square_size),
                                     square_size, square_size,
                                     linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
    ax.text(input_origin[0] + (cols / 2) * square_size, input_origin[1] + 0.5, "Input",
            ha='center', fontsize=12)

    # 在输入矩阵上标示卷积核作用区域（3x3区域）
    kernel_origin = (input_origin[0], input_origin[1])
    kernel_size = 3
    rect_kernel = patches.Rectangle(kernel_origin, kernel_size * square_size, kernel_size * square_size,
                                    linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect_kernel)
    ax.text(kernel_origin[0] + kernel_size / 2, kernel_origin[1] - kernel_size / 2, "Kernel",
            color='red', ha='center', va='center', fontsize=10)

    # 绘制卷积运算箭头
    arrow_start = (input_origin[0] + kernel_size * square_size + 0.5, input_origin[1] - kernel_size / 2)
    arrow_end = (input_origin[0] + kernel_size * square_size + 2, input_origin[1] - kernel_size / 2)
    ax.annotate("", xy=arrow_end, xytext=arrow_start, arrowprops=dict(arrowstyle="->", lw=2))
    ax.text((arrow_start[0] + arrow_end[0]) / 2, arrow_start[1] + 0.3, "Convolution", ha='center', fontsize=10)

    # 绘制输出单元（卷积结果）
    output_origin = (input_origin[0] + kernel_size * square_size + 3, input_origin[1] - 1)
    output_square = patches.Rectangle(output_origin, square_size, square_size,
                                      linewidth=1, edgecolor='black', facecolor='lightblue')
    ax.add_patch(output_square)
    ax.text(output_origin[0] + square_size / 2, output_origin[1] + square_size / 2, "Output",
            ha='center', va='center', fontsize=10)

    # 设置图像边界和隐藏坐标轴
    ax.set_xlim(-1, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    plt.title("2D Convolution Operation")
    plt.tight_layout()

    # 保存为高清图片
    plt.savefig("2d_convolution_diagram.png", dpi=300)
    plt.show()


draw_conv_diagram()
