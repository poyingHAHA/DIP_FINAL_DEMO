#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:59:21 2024

@author: lanweiren
"""
import os
import sys
sys.path.append('/Users/lanweiren/Downloads/automatic-watermark-detection-master/src')
import cv2
from src import *
from src.watermark_reconstruct import restoreImages

def plot(img):
    plt.imshow(img.astype(np.uint8))
    plt.show()

# gradient取中位數得watermark gradient
gx, gy, gxlist, gylist = estimate_watermark('watermarked')

# watermark gradient夠大的地方才視為watermark的範圍
# 對該範圍做bounding box加上一點邊界寬度，擷取出watermark的範圍
cropped_gx, cropped_gy = crop_watermark(gx, gy, 0.5)
# 找到浮水印的pattern，重建為W_m，但要shift之後才是真的Wm
W_m = poisson_reconstruct(cropped_gx, cropped_gy)
Wm = W_m - W_m.min() # 平移最小值為0，之後會再shift ∇α·E[Ik]，所以這裡仍不是真正的Wm
# 記得Wm = α * W，Ｗm是estimation of matted Watermark

# 這裡之後改成每張圖各做一次，處理浮水印在不同位置的情況
img = cv2.imread('watermarked/fotolia_137840668.jpg')
# Chamfer distance
im, start, end = watermark_detector(img, cropped_gx, cropped_gy, 200, 220)


plot(im)



# 算出K
K = len(gxlist)

# 擷取每張圖片中有浮水印得部分J
J, img_paths = get_cropped_images('watermarked', K, start, end, cropped_gx.shape)

# get threshold of W_m for alpha matte estimate
# 每張圖每個pixel都有三通道α值，我們可以直接算出αn，並先設定三通道一模一樣，照理說不會有太大差異，所以是最佳化不錯的初始設定
alph_est = estimate_normalized_alpha(J, Wm, K) # 得到αn
alph = np.stack([alph_est, alph_est, alph_est], axis=2) # αn 三通道值一樣

# 求C, blending factor, 一通道一個
#C, est_Ik = estimate_blend_factor(J, Wm, alph,2.55)
C, est_Ik = estimate_blend_factor2(J, Wm, alph,2.55)

# αn * C = α
alpha = alph.copy()
for i in range(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]

# shift α·E[Ik]
Wm = Wm + alpha*est_Ik

# 三個通道除以C得浮水印本身：Ｗ，也就是blending factor=1的樣子
W = Wm.copy()
for i in range(3):
	W[:,:,i]/=C[i]

# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(J, W_m, alpha, W, iters=2)
# Ik是我們要的結果

restored_Ik = restoreImages(img_paths, Ik, start, end)
plot(restored_Ik[0])
image_rgb = cv2.cvtColor(restored_Ik[0], cv2.COLOR_BGR2RGB)
plot(image_rgb)


output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the list of images
for i, img in enumerate(restored_Ik):
    # Convert from BGR to RGB
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Construct the output filename
    output_filename = os.path.join(output_dir, f'image2_{i}.png')
        
    # Save the image in the specified folder as PNG
    cv2.imwrite(output_filename, img)
    

def calculate_psnr_opencv(original_images, restored_images):
    psnr_values = []

    for original, restored in zip(original_images, restored_images):
        # Check if both images have the same dimensions
        if original.shape != restored.shape:
            print("Error: Image dimensions do not match.")
            print(original.shape)
            print(restored.shape)
            psnr_values.append(None)
            continue
        
        # Use OpenCV's built-in function to calculate PSNR
        psnr = cv2.PSNR(original, restored)
        psnr_values.append(psnr)

    return psnr_values

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):  # 使用 sorted 保證檔案順序
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 確保只處理圖像檔案
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
    return images

# 設定圖像資料夾路徑
original_images_folder = 'final'
restored_images_folder = 'output'

# 加載圖像
original_images = load_images_from_folder(original_images_folder)
restored_images = load_images_from_folder(restored_images_folder)

psnr_values1 = calculate_psnr_opencv(original_images[:11], restored_images[:11])
psnr_values2 = calculate_psnr_opencv(original_images[:11], restored_images[11:22])

mean1 = np.mean(psnr_values1)
mean2 = np.mean(psnr_values2)
std1 = np.std(psnr_values1)
std2 = np.std(psnr_values2)

print(f"Method 1: Mean PSNR = {mean1}, Standard Deviation = {std1}")
print(f"Method 2: Mean PSNR = {mean2}, Standard Deviation = {std2}")

if mean1 > mean2:
    print("Method 1 has a higher average PSNR.")
elif mean1 < mean2:
    print("Method 2 has a higher average PSNR.")
else:
    print("Both methods have the same average PSNR.")







