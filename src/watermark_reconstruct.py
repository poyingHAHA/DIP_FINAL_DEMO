import numpy as np
import cv2
import os
import scipy
from scipy.sparse import *
from scipy.sparse import linalg
from estimate_watermark import *
from closed_form_matting import *
from numpy import nan, isnan

import matplotlib.pyplot as plt
from scipy.sparse import diags, vstack, hstack
from scipy.sparse.linalg import spsolve

def get_cropped_images(foldername, num_images, start, end, shape):
    '''
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    '''
    images_cropped = np.zeros((num_images,) + shape)
    # get images
    # Store all the watermarked images
    # start, and end are already stored
    # just crop and store image
    image_paths = []
    _s, _e = start, end
    index = 0

    # Iterate over all images
    for r, dirs, files in os.walk(foldername):

        for file in files:
            _img = cv2.imread(os.sep.join([r, file]))
            if _img is not None:
                # estimate the watermark part
                image_paths.append(os.sep.join([r, file]))
                _img = _img[_s[0]:(_s[0]+_e[0]), _s[1]:(_s[1]+_e[1]), :]
                # add to list images
                images_cropped[index, :, :, :] = _img
                index+=1
            else:
                print("%s not found."%(file))

    return (images_cropped, image_paths)

# lan 擷取多張圖
def get_cropped_images2(ims, starts, ends):
    cropped_images = []
    for i in range(len(ims)):
        start_y, start_x = starts[i]
        end_y, end_x = ends[i]
        
        # 使用 start 和 end 坐標來裁剪圖片
        cropped_img = ims[i][start_y:end_y, start_x:end_x]
        cropped_images.append(cropped_img)
    
    # 將裁剪後的圖片列表轉換為 NumPy array
    cropped_images_array = np.array(cropped_images)
    
    return cropped_images_array


# get sobel coordinates for y
def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i-1, j, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i+1, j, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# get sobel coordinates for x
def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i, j-1, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i, j+1, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# filter
def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i>=0 and i<m and j>=0 and j<n:
        return True

# Change to ravel index
# also filters the wrong guys
def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)

# TODO: Consider wrap around of indices to remove the edge at the end of sobel
# get Sobel sparse matrix for Y
def get_ySobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get Sobel sparse matrix for X
def get_xSobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))

# get estimated normalized alpha matte
def estimate_normalized_alpha(J, W_m, num_images=30, threshold=170, invert=False, adaptive=False, adaptive_threshold=21, c2=10):
    # 浮水印轉灰階，用三通道求平均而得，確保轉為uint8形式
    _Wm = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
    if adaptive:
        # 二值化也可以不用自己設定threshold
        thr = cv2.adaptiveThreshold(_Wm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_threshold, c2)
    else:
        # 二值化為thr, 閾值為ret也就是170
        ret, thr = cv2.threshold(_Wm, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        thr = 255-thr
    thr = np.stack([thr, thr, thr], axis=2)

    num, m, n, p = J.shape
    alpha = np.zeros((num_images, m, n))
    iterpatch = 900 # 沒用到

    print("Estimating normalized alpha using %d images."%(num_images))
    # for all images, calculate alpha
    for idx in range(num_images):
        imgcopy = thr
        # 
        alph = closed_form_matte(J[idx], imgcopy)
        alpha[idx] = alph
    
    # 每張圖算出估計的αn, 取中位數, 浮水印mask內才會有透明度最佳化
    alpha = np.median(alpha, axis=0)
    return alpha

def estimate_blend_factor(J, W_m, alph, threshold=0.01*255):
    '''
    所有J大致去浮水印得Jm
    計算所有Jm的梯度得Jm_grad
    計算estIk_grad
    在black patch用least squares算c
    '''
    K, m, n, p = J.shape
    # Jm為未精確去浮水印的bounding box
    Jm = (J - W_m)
    gx_jm = np.zeros(J.shape)
    gy_jm = np.zeros(J.shape)

    for i in range(K):
        gx_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 1, 0, 3)
        gy_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 0, 1, 3)

    Jm_grad = np.sqrt(gx_jm**2 + gy_jm**2)
    #  est_Ik = αn * E[Ik] 而非E[Ik]
    est_Ik = alph*np.median(J, axis=0)
    gx_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 1, 0, 3)
    gy_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 0, 1, 3)
    estIk_grad = np.sqrt(gx_estIk**2 + gy_estIk**2)

    C = []
    for i in range(3):
        # 在black patch用least squares算c
        c_i = np.sum(Jm_grad[:,:,:,i]*estIk_grad[:,:,i])/np.sum(np.square(estIk_grad[:,:,i]))/K
        print(c_i)
        C.append(c_i)

    return C, est_Ik

# 嘗試照論文方法找C
def estimate_blend_factor2(J, W_m, alph, threshold=0.01*255):
    '''
    所有J大致去浮水印得Jm
    計算所有Jm的梯度得Jm_grad
    計算estIk_grad
    在black patch用least squares算c
    '''
    K, m, n, p = J.shape
    # Jm為未精確去浮水印的bounding box
    Jm = (J - W_m)
    gx_jm = np.zeros(J.shape)
    gy_jm = np.zeros(J.shape)

    for i in range(K):
        gx_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 1, 0, 3)
        gy_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 0, 1, 3)

    Jm_grad = np.sqrt(gx_jm**2 + gy_jm**2)
    #  (E[Ik]) est_Ik = αn * E[Ik] 而非E[Ik]
    # We estimate DC = E[Ik] as the median instensity across the image collection at each Ik'th patch location, and plug our initial estimation of αn
    est_Ik = alph*np.median(J, axis=0)
    gx_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 1, 0, 3)
    gy_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 0, 1, 3)
    estIk_grad = np.sqrt(gx_estIk**2 + gy_estIk**2)

    C = []
    for i in range(3):
        # 在black patch用least squares算c
        # 對於每個像素點，Jm梯度和E[Ik]梯度越一致，代表這個點像素質幾乎由浮水印決定，blending factor越高, (mask外相乘必為0)
        # 我們其實不是特意取找某個黑點，而是較黑的浮水印影響較多，較亮的浮水印影響較少，最後做最小二乘法平均
        # 畢竟黑點不是一定有，(通常有但有時輸入圖片數量較小)
        numerator = 0
        denominator = 0
        for k in range(K):
            for x in range(m):
                for y in range(n):
                    if Jm[k, x, y, i] <= threshold:
                        numerator += Jm_grad[k, x, y, i] * estIk_grad[x, y, i]
                        denominator += estIk_grad[x, y, i] ** 2
        # 不知道為何，不用除以K才正確
        c_i = numerator / denominator
        print(c_i)
        C.append(c_i)
    
    return C, est_Ik


def Func_Phi(X, epsilon=1e-3):
    return np.sqrt(X + epsilon**2)

def Func_Phi_deriv(X, epsilon=1e-3):
    return 0.5/Func_Phi(X, epsilon)

# IRLS最佳化，就是普通最佳化加上權重而已
def solve_images(J, W_m, alpha, W_init, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=4):
    '''
    Master solver, follows the algorithm given in the supplementary.
    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    '''
    # prepare variables
    K, m, n, p = J.shape
    size = m*n*p

    sobelx = get_xSobel_matrix(m, n, p)
    sobely = get_ySobel_matrix(m, n, p)
    Ik = np.zeros(J.shape)
    Wk = np.zeros(J.shape)
    for i in range(K):
        Ik[i] = J[i] - W_m
        Wk[i] = W_init.copy()

    # This is for median images
    W = W_init.copy()

    # Iterations
    for _ in range(iters):

        print("------------------------------------")
        print("Iteration: %d"%(_))

        # Step 1，固定W, alpha 對Wk, Ik最佳化
        # IRLS將非線性優化問題轉換為線性優化問題
        print("Step 1")
        alpha_gx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, 3)
        alpha_gy = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, 3)

        Wm_gx = cv2.Sobel(W_m, cv2.CV_64F, 1, 0, 3)
        Wm_gy = cv2.Sobel(W_m, cv2.CV_64F, 0, 1, 3)

        cx = diags(np.abs(alpha_gx).reshape(-1))
        cy = diags(np.abs(alpha_gy).reshape(-1))

        alpha_diag = diags(alpha.reshape(-1))
        alpha_bar_diag = diags((1-alpha).reshape(-1))
        # K張圖分別最佳化
        for i in range(K):
            # prep vars
            Wkx = cv2.Sobel(Wk[i], cv2.CV_64F, 1, 0, 3)
            Wky = cv2.Sobel(Wk[i], cv2.CV_64F, 0, 1, 3)

            Ikx = cv2.Sobel(Ik[i], cv2.CV_64F, 1, 0, 3)
            Iky = cv2.Sobel(Ik[i], cv2.CV_64F, 0, 1, 3)

            alphaWk = alpha*Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)   
            
            # 根據各項定義，求導數，建對角矩陣，為權重矩陣
            # 數據項Edata: Edata(Ik,Wk)=  ||αWk+(1-α)Ik-Jk||^2
            phi_data = diags( Func_Phi_deriv(np.square(alpha*Wk[i] + (1-alpha)*Ik[i] - J[i]).reshape(-1)) )
            # 正則化項Ereg(∇Wk) = ||∇Wk||^2
            #phi_W = diags( Func_Phi_deriv(np.square( np.abs(alpha_gx)*Wkx + np.abs(alpha_gy)*Wky  ).reshape(-1)) )
            # 正則化項Ereg(∇Ik) = ||∇Ik||^2
            #phi_I = diags( Func_Phi_deriv(np.square( np.abs(alpha_gx)*Ikx + np.abs(alpha_gy)*Iky  ).reshape(-1)) )
            # 保真項Ef(∇(αWk)) = ||∇(αWk) - ∇Wm||^2
            phi_f = diags( Func_Phi_deriv( ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2 ).reshape(-1)) )
            # 輔助項Eaux(W,Wk) = ||Wk-W||^2
            phi_aux = diags( Func_Phi_deriv(np.square(Wk[i] - W).reshape(-1)) )
            # 正則化項Ereg(∇Ik) = ||∇Ik||^2 梯度正則化
            phi_rI = diags( Func_Phi_deriv( np.abs(alpha_gx)*(Ikx**2) + np.abs(alpha_gy)*(Iky**2) ).reshape(-1) )
            # 正則化項Ereg(∇Wk) = ||∇Wk||^2 梯度正則化
            phi_rW = diags( Func_Phi_deriv( np.abs(alpha_gx)*(Wkx**2) + np.abs(alpha_gy)*(Wky**2) ).reshape(-1) )
            
            # 構建L matrix
            # L_i = Sx^T(Cx*phi_rI)Sx + Sy^T(Cx*phi_rI)Sy, Cx, Cy是对角矩阵，表示alpha Sobel梯度的绝对值。用于调整梯度的权重
            L_i = sobelx.T.dot(cx*phi_rI).dot(sobelx) + sobely.T.dot(cy*phi_rI).dot(sobely)
            # L_w = Sx^T(Cx*phi_rW)Sx + Sy^T(Cx*phi_rW)Sy
            L_w = sobelx.T.dot(cx*phi_rW).dot(sobelx) + sobely.T.dot(cy*phi_rW).dot(sobely)
            # L_f = Sx^T(Cx*phi_f)Sx + Sy^T(Cx*phi_f)Sy
            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            # A_f = alpha_diag^T * L_f * alpha_diag + r * phi_aux
            A_f = alpha_diag.T.dot(L_f).dot(alpha_diag) + gamma*phi_aux
            # b
            bW = alpha_diag.dot(phi_data).dot(J[i].reshape(-1)) + beta*L_f.dot(W_m.reshape(-1)) + gamma*phi_aux.dot(W.reshape(-1))
            bI = alpha_bar_diag.dot(phi_data).dot(J[i].reshape(-1))
            # A, 用L matrix構建
            A = vstack([hstack([(alpha_diag**2)*phi_data + lambda_w*L_w + beta*A_f, alpha_diag*alpha_bar_diag*phi_data]), \
                         hstack([alpha_diag*alpha_bar_diag*phi_data, (alpha_bar_diag**2)*phi_data + lambda_i*L_i])]).tocsr()
            # 解Ax=b
            b = np.hstack([bW, bI])
            # 稀疏矩陣求解x=A^-1 b
            x = linalg.spsolve(A, b)
            # 更新Wk, Ik
            Wk[i] = x[:size].reshape(m, n, p)
            Ik[i] = x[size:].reshape(m, n, p)
            # Ik[i]轉回uint8
            #output_image = (Ik[i] * 255).astype(np.uint8) if Ik[i].max() <= 1.0 else Ik[i].astype(np.uint8)
        
            # Construct the filename and path to save the image
            #output_path = os.path.join('output', f"output_image_{i}.png")
        
            # Save the image
            #cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV
            plt.subplot(3,1,1); plt.imshow(PlotImage(J[i]))
            plt.subplot(3,1,2); plt.imshow(PlotImage(Wk[i]))
            plt.subplot(3,1,3); plt.imshow(PlotImage(Ik[i]))
            plt.draw()
            plt.pause(0.001)
            print(i)

        # Step 2: 各個Wk更新後，取中位數就可以直接最佳化W, 因為W-Wk是penalty
        print("Step 2")
        W = np.median(Wk, axis=0)

        plt.imshow(PlotImage(W))
        plt.draw()
        plt.pause(0.001)
        
        # Step 3: 最佳化α
        print("Step 3")
        W_diag = diags(W.reshape(-1))
        
        # 一樣K張圖分別做
        for i in range(K):
            alphaWk = alpha*Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)
            # 保真項Ef(∇(αW)) = ||∇(αW) - ∇W||^2
            phi_f = diags( Func_Phi_deriv( ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2 ).reshape(-1)) )
            # 數據項Edata(α,Ik,W)，權重
            phi_kA = diags(( (Func_Phi_deriv((((alpha*Wk[i] + (1-alpha)*Ik[i] - J[i])**2)))) * ((W-Ik[i])**2)  ).reshape(-1))
            phi_kB = (( (Func_Phi_deriv((((alpha*Wk[i] + (1-alpha)*Ik[i] - J[i])**2))))*(W-Ik[i])*(J[i]-Ik[i])  ).reshape(-1))
            # 正則化項Ereg(∇α) = ||∇α||^2 梯度正則化
            phi_alpha = diags(Func_Phi_deriv(alpha_gx**2 + alpha_gy**2).reshape(-1))
            
            # L matrix計算
            # L_f = Sx^T(phi_alpha)Sx + Sy^T(phi_alpha)Sy
            L_alpha = sobelx.T.dot(phi_alpha.dot(sobelx)) + sobely.T.dot(phi_alpha.dot(sobely))
            # L_f = Sx^T(phi_f)Sx + Sy^T(phi_f)Sy
            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            A_tilde_f = W_diag.T.dot(L_f).dot(W_diag)
            # K張圖加總算出A, b
            if i==0: # 第一張圖
                A1 = phi_kA + lambda_a*L_alpha + beta*A_tilde_f
                b1 = phi_kB + beta*W_diag.dot(L_f).dot(W_m.reshape(-1))
            else:
                A1 += (phi_kA + lambda_a*L_alpha + beta*A_tilde_f)
                b1 += (phi_kB + beta*W_diag.T.dot(L_f).dot(W_m.reshape(-1)))
        # 解稀疏矩陣，Ax = b
        alpha = linalg.spsolve(A1, b1).reshape(m,n,p)

        plt.imshow(PlotImage(alpha))
        plt.draw()
        plt.pause(0.001)
    
    return (Wk, Ik, W, alpha)


def changeContrastImage(J, I):
    cJ1 = J[0, 0, :]
    cJ2 = J[-1, -1, :]

    cI1 = I[0, 0, :]
    cI2 = I[-1,-1, :]

    I_m = cJ1 + (I-cI1)/(cI2-cI1)*(cJ2-cJ1)
    return I_m

def restoreImages(img_paths, Ik, start, end):
    restored_images = []
    y_start, x_start = start
    height, width = end

    # Loop through each image path and corresponding Ik
    for path, ik_patch in zip(img_paths, Ik):
        # Read the original image
        original_image = cv2.imread(path)
        
        # Check if the image is loaded properly
        if original_image is None:
            print(f"Failed to load image from {path}")
            continue

        # Calculate the end positions for the patch
        x_end = x_start + width
        y_end = y_start + height

        # Replace the specified portion of the original image with the corresponding Ik patch
        original_image[y_start:y_end, x_start:x_end] = ik_patch

        # Append the modified image to the list
        restored_images.append(original_image)

    return restored_images
    
def restoreImages_dif_loc(imgs, Ik, starts, ends):
    restored_images = []

    # Loop through each set of image, Ik patch, start and end coordinates
    for img, ik_patch, start, end in zip(imgs, Ik, starts, ends):
        # Unpack start and end points
        x_start, y_start = start
        x_end, y_end = end

        # Check if the patches and regions are valid
        if x_end > img.shape[0] or y_end > img.shape[1]:
            print("Patch end point is outside the image dimensions.")
            continue
        
        # Replace the specified portion of the image with the corresponding Ik patch
        img[x_start:x_end, y_start:y_end] = ik_patch

        # Append the modified image to the list
        restored_images.append(img)

    return restored_images 
    
    
    
    
    
    