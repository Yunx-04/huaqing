import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw(image):
    # 可视化RGB图像
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.title('image')
    plt.axis('off')
    plt.show()


# 调整亮度
def adjust_brightness(image, target_brightness):
    # 计算当前图片的平均亮度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    print("当前亮度:", current_brightness)
    # 计算亮度调整因子
    brightness_factor = target_brightness / current_brightness
    # 调整亮度
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image


# 添加饱和度
def adjust_saturation(img, factor):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 分离H,S,V通道
    h, s, v = cv2.split(img_hsv)
    # 调整饱和度（S通道）
    # 将S通道的值乘以因子，并确保值在0-255的范围内
    s_adjusted = np.clip(s * factor, 0, 255).astype(np.uint8)
    # 合并调整后的S通道与其他通道
    img_hsv_adjusted = cv2.merge([h, s_adjusted, v])
    # 将HSV图片转换回BGR颜色空间
    img_bar_output = cv2.cvtColor(img_hsv_adjusted, cv2.COLOR_HSV2BGR)
    return img_bar_output


def shuchu(tif_file):
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段（假设波段顺序为B02, B03, B04, B08, B12）
        bands = src.read()  # 形状为 (波段数, 高度, 宽度)，这里是 (5, height, width)

    # 归一化所有波段到0-255范围
    bands_normalized = (bands / 10000.0 * 255.0).astype(np.uint8)

    # 提取各波段
    blue = bands_normalized[0]  # B02 - 蓝
    green = bands_normalized[1]  # B03 - 绿
    red = bands_normalized[2]  # B04 - 红
    nir = bands_normalized[3]  # B08 - 近红外
    swir = bands_normalized[4]  # B12 - 短波红外

    # 真彩色合成（RGB）
    rgb_image = np.dstack((red, green, blue))

    # 调整亮度
    img_adjusted = adjust_brightness(rgb_image, 100)

    # 输出真彩色图像
    draw(img_adjusted)

    # 假彩色合成（近红外、红、绿）
    false_color = np.dstack((nir, red, green))

    # 调整亮度
    false_color_adjusted = adjust_brightness(false_color, 100)

    # 输出假彩色图像
    draw(false_color_adjusted)

    # 返回结果
    return img_adjusted, false_color_adjusted


if __name__ == '__main__':
    # 替换为您的TIFF文件路径
    tif_file_path = "2019_1101_nofire_B2348_B12_10m_roi.tif"

    # 处理图像
    true_color, false_color = shuchu(tif_file_path)