#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:00:31 2024

@author: lanweiren
"""
import os
import sys
sys.path.append('/Users/lanweiren/Downloads/automatic-watermark-detection-master/src')
import cv2
from src import *
from src.gui import get_matted_watermark
from src.watermark_reconstruct import get_cropped_images2
from src.watermark_reconstruct import restoreImages_dif_loc
from src.estimate_watermark import estimate_watermark2
from src.estimate_watermark import estimate_watermark_from_J

def plot(img):
    plt.imshow(img.astype(np.uint8))
    plt.show()

poisson_reconstruct_grad = False

########## step1: Estimate the watermark ##########
    
# 執行GUI程式，透過GUI介面取得matted watermark，可以一張或多張，越多張越準確
matted_watermarks = get_matted_watermark()
    
# 透過estimate_watermark函數計算出原始水印可能的梯度
matted_watermarks_arr = [np.array(matted_watermark) for matted_watermark in matted_watermarks]
(Wm_x, Wm_y, gx_crops, gy_crops) = estimate_watermark2(matted_watermarks_arr)
plot(Wm_x)
# 透過poisson_reconstruct函數計算出原始水印
W_m, _ = poisson_reconstruct3(Wm_x, Wm_y)
Wm = W_m - W_m.min()
plot(Wm)
# 經過poisson_reconstruct函數重建的水印圖像，再做一次梯度計算(可能)可以得到更準確的梯度
if poisson_reconstruct_grad:
    Wm_x = cv2.Sobel(Wm, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
    Wm_y = cv2.Sobel(Wm, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)

########## step2: Watermark Detection ##########
# 重建水印圖像的x與y方向梯度
gx_est = cv2.Sobel(Wm, cv2.CV_64F, 1, 0, ksize=1)
gy_est = cv2.Sobel(Wm, cv2.CV_64F, 0, 1, ksize=1)
Wm2, _ = poisson_reconstruct3(gx_est, gy_est)
Wm2 = Wm2 - Wm2.min()


# 設定圖片資料夾的路徑
folder_path = './mountain_real/watermarked_alpha2/'
# 獲取資料夾中所有文件的名稱
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
# 初始化列表，用於儲存每張圖片的處理結果
ims = []
starts = []
ends = []

# 處理每一張圖片
for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 假設 watermark_detector 是一個定義好的函數，並且 gx_est, gy_est 已經被定義
    im, start, end = watermark_detector2(img, gx_est, gy_est, 100, 200)
    
    # 儲存每張圖片的處理結果
    ims.append(im)
    starts.append(start)
    ends.append(end)

# 算出K
K = len(ims)

# 擷取每張圖片中有浮水印得部分J
J = get_cropped_images2(ims, starts, ends)
gx, gy, gxlist, gylist = estimate_watermark_from_J(J)
#cropped_gx, cropped_gy = crop_watermark(gx, gy, 0.5)
W_m = poisson_reconstruct(gx, gy)
Wm = W_m - W_m.min() # 平移最小值為0，之後會再shift ∇α·E[Ik]，所以這裡仍不是真正
plot(Wm)
for i in J:
    plot(i)


########## step3: watermark decomposition ##########

# get threshold of W_m for alpha matte estimate
# 每張圖每個pixel都有三通道α值，我們可以直接算出αn，並先設定三通道一模一樣，照理說不會有太大差異，所以是最佳化不錯的初始設定
alph_est = estimate_normalized_alpha(J, Wm, K) # 得到αn
alph = np.stack([alph_est, alph_est, alph_est], axis=2) # αn 三通道值一樣

# 求C, blending factor, 一通道一個
C, est_Ik = estimate_blend_factor(J, Wm, alph,2.55)
#C, est_Ik = estimate_blend_factor2(J, Wm, alph,2.55)

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
Wk, Ik, W, alpha1 = solve_images(J, W_m, alpha, W, iters=1)
# Ik是我們要的結果
plot(Ik[34])
restored_Ik = restoreImages_dif_loc(ims, Ik, starts, ends)
plot(restored_Ik[34])
image_rgb = cv2.cvtColor(restored_Ik[0], cv2.COLOR_BGR2RGB)
plot(image_rgb)


output_dir = 'output_dif_loc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the list of images
for i, img in enumerate(restored_Ik):
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Construct the output filename
    output_filename = os.path.join(output_dir, f'image_{i}.png')
        
    # Save the image in the specified folder as PNG
    cv2.imwrite(output_filename, rgb_image)
