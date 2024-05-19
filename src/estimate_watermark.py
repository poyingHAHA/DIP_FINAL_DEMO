import sys, os
import cv2
import numpy as np
import warnings
from matplotlib import pyplot as plt
import math
import numpy
import scipy, scipy.fftpack

# Variables
KERNEL_SIZE = 3

# po ying修正版
def estimate_watermark2(images):
    """
    Given a folder, estimate the watermark (grad(W) = median(grad(J)))
    Also, give the list of gradients, so that further processing can be done on it
    """
    if not images:
        warnings.warn("No images found in the folder.", UserWarning)
        return None

    # Compute gradients
    print("Computing gradients.")
    gradx = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), images))
    grady = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), images))
    
    # 以第一張圖片為基準，對齊其他圖片，所以第一章圖片一定要最小
    gx_crop = [gradx[0]]
    gy_crop = [grady[0]]
    for i in range(1,len(gradx)):
      im, start, end = watermark_detector2(images[i], gradx[0], grady[0])
      gx_crop.append(gradx[i][start[0]:end[0], start[1]:end[1], :])
      gy_crop.append(grady[i][start[0]:end[0], start[1]:end[1], :])
      print(start, end)
    

    # Compute median of grads
    print("Computing median gradients.")
    Wm_x = np.median(np.array(gx_crop), axis=0)
    Wm_y = np.median(np.array(gy_crop), axis=0)

    return (Wm_x, Wm_y, gx_crop, gy_crop)

# 原版
def estimate_watermark(foldername):
    """
    Given a folder, estimate the watermark (grad(W) = median(grad(J)))
    Also, give the list of gradients, so that further processing can be done on it
    """
    if not os.path.exists(foldername):
        warnings.warn("Folder does not exist.", UserWarning)
        return None

    images = []
    for r, dirs, files in os.walk(foldername):
        # Get all the images
        for file in files:
            img = cv2.imread(os.sep.join([r, file]))
            if img is not None:
                images.append(img)
            else:
                print("%s not found." % (file))

    if not images:
        warnings.warn("No images found in the folder.", UserWarning)
        return None

    # Compute gradients
    print("Computing gradients.")
    gradx = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), images))
    grady = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), images))

    # Compute median of grads
    print("Computing median gradients.")
    Wm_x = np.median(np.array(gradx), axis=0)                    
    Wm_y = np.median(np.array(grady), axis=0)

    return (Wm_x, Wm_y, gradx, grady)

def estimate_watermark_from_J(J):
    # Compute gradients
    print("Computing gradients.")
    gradx = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), J))
    grady = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), J))

    # Compute median of grads
    print("Computing median gradients.")
    Wm_x = np.median(np.array(gradx), axis=0)                    
    Wm_y = np.median(np.array(grady), axis=0)

    return (Wm_x, Wm_y, gradx, grady)

def PlotImage(image):
	""" 
	PlotImage: Give a normalized image matrix which can be used with implot, etc.
	Maps to [0, 1]
	"""
	im = image.astype(float)
	return (im - np.min(im))/(np.max(im) - np.min(im))


def poisson_reconstruct2(gradx, grady, boundarysrc):
	# Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
	# Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

	# Laplacian
	gyy = grady[1:,:-1] - grady[:-1,:-1]
	gxx = gradx[:-1,1:] - gradx[:-1,:-1]
	f = numpy.zeros(boundarysrc.shape)
	f[:-1,1:] += gxx
	f[1:,:-1] += gyy

	# Boundary image
	boundary = boundarysrc.copy()
	boundary[1:-1,1:-1] = 0;

	# Subtract boundary contribution
	f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
	f = f[1:-1,1:-1] - f_bp

	# Discrete Sine Transform
	tt = scipy.fftpack.dst(f, norm='ortho')
	fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

	# Eigenvalues
	(x,y) = numpy.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
	denom = (2*numpy.cos(math.pi*x/(f.shape[1]+2))-2) + (2*numpy.cos(math.pi*y/(f.shape[0]+2)) - 2)

	f = fsin/denom

	# Inverse Discrete Sine Transform
	tt = scipy.fftpack.idst(f, norm='ortho')
	img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

	# New center + old boundary
	result = boundary
	result[1:-1,1:-1] = img_tt

	return result

# po ying修正版
def poisson_reconstruct3(gradx, grady, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1, 
		boundary_image=None, boundary_zero=True):
	"""
	Iterative algorithm for Poisson reconstruction. 
	Given the gradx and grady values, find laplacian, and solve for image
	Also return the squared difference of every step.
	h = convergence rate
	"""
	# 計算二階微分
	fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)
	fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # 透過二階微分計算原圖的laplacian
	laplacian = fxx + fyy
	m,n,p = laplacian.shape

	# 初始化重建圖像est，如果boundary_zero為True，則初始化為0，表重建時不考慮邊界對重建的影響
 	# 如果boundary_image不為None，則初始化為boundary_image ，表重建時考慮邊界對重建的影響
	if boundary_zero == True:
		est = np.zeros(laplacian.shape)
	else:
		assert(boundary_image is not None)
		assert(boundary_image.shape == laplacian.shape)
		est = boundary_image.copy()

	# 將內部的值初始化為0~1隨機值，並保持邊界不變
	est[1:-1, 1:-1, :] = np.random.random((m-2, n-2, p))
	loss = []

	# 迭代求解
	for i in range(num_iters):
		old_est = est.copy()
		# 取上下左右四個點的平均值，減去h*h*laplacian，得到新的est
		est[1:-1, 1:-1, :] = 0.25*(est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :] - h*h*laplacian[1:-1, 1:-1, :])
		# 計算誤差，預期重建圖像est應該與上一次迭代的est差異越來越小
		error = np.sum(np.square(est-old_est))
		loss.append(error)

	return (est, loss)

# 原版
def poisson_reconstruct(gradx, grady, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1, 
		boundary_image=None, boundary_zero=True):
	"""
	Iterative algorithm for Poisson reconstruction. 
	Given the gradx and grady values, find laplacian, and solve for image
	Also return the squared difference of every step.
	h = convergence rate
	"""
	fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)
	fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)
	laplacian = fxx + fyy
	m,n,p = laplacian.shape

	if boundary_zero == True:
		est = np.zeros(laplacian.shape)
	else:
		assert(boundary_image is not None)
		assert(boundary_image.shape == laplacian.shape)
		est = boundary_image.copy()

	est[1:-1, 1:-1, :] = np.random.random((m-2, n-2, p))
	loss = []

	for i in range(num_iters):
		old_est = est.copy()
		est[1:-1, 1:-1, :] = 0.25*(est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :] - h*h*laplacian[1:-1, 1:-1, :])
		error = np.sum(np.square(est-old_est))
		loss.append(error)

	return (est)


def image_threshold(image, threshold=0.5):
	'''
	Threshold the image to make all its elements greater than threshold*MAX = 1
	'''
	m, M = np.min(image), np.max(image)
	im = PlotImage(image)
	im[im >= threshold] = 1
	im[im < 1] = 0
	return im


def crop_watermark(gradx, grady, threshold=0.4, boundary_size=2):
	"""
	Crops the watermark by taking the edge map of magnitude of grad(W)
	Assumes the gradx and grady to be in 3 channels
	@param: threshold - gives the threshold param
	@param: boundary_size - boundary around cropped image
	"""
	W_mod = np.sqrt(np.square(gradx) + np.square(grady))
    # normalize to [0,1]
	W_mod = PlotImage(W_mod)
    # set pixel to 0 or 1
	W_gray = image_threshold(np.average(W_mod, axis=2), threshold=threshold)
	x, y = np.where(W_gray == 1)
    
    # bounding box再寬2+1 pixels
	xm, xM = np.min(x) - boundary_size - 1, np.max(x) + boundary_size + 1
	ym, yM = np.min(y) - boundary_size - 1, np.max(y) + boundary_size + 1

	return gradx[xm:xM, ym:yM, :] , grady[xm:xM, ym:yM, :]


def normalized(img):# [0,1]變成[-1,1]
	"""
	Return the image between -1 to 1 so that its easier to find out things like 
	correlation between images, convolutionss, etc.
	Currently required for Chamfer distance for template matching.
	"""
	return (2*PlotImage(img)-1)

# po ying修正版
def watermark_detector2(img, gx, gy, thresh_low=200, thresh_high=220, printval=False, draw_rect=True):
    """
    Compute a verbose edge map using Canny edge detector, take its magnitude.
    Assuming cropped values of gradients are given.
    Returns image, start and end coordinates
    """
    Wm = np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2)

	# 如果用Canny再做chamfer distance，可能會因為原本圖片的邊緣太多，導致找不到水印的位置
    img_edgemap = cv2.Canny(img, thresh_low, thresh_high)
    chamfer_dist = cv2.filter2D(img_edgemap.astype(float), -1, Wm, borderType=cv2.BORDER_CONSTANT)

    rect = Wm.shape
    index = np.unravel_index(np.argmax(chamfer_dist), img.shape[:-1])
    if printval:
        print(index)

    # 將x, y座標轉換為左上角座標
    x, y = int(index[0]-rect[0]/2), int(index[1]-rect[1]/2)
    # Calculate the coordinates of the bottom-right corner of the rectangle
    x_end, y_end = int(x + rect[0]), int(y + rect[1])
    
    im = img.copy()
    # Draw rectangle on image
    if draw_rect:
        cv2.rectangle(im, (y, x), (y + rect[1], x + rect[0]), (255, 0, 0), 3)  # 加粗以更清晰显示
    
    # return (im, (x, y), (rect[0], rect[1]), (x_end, y_end))
    return (im, (x, y), (x_end, y_end))

# 原始
def watermark_detector(img, gx, gy, thresh_low=200, thresh_high=220, printval=False):
    """
    Compute a verbose edge map using Canny edge detector, take its magnitude.
    Assuming cropped values of gradients are given.
    Returns image, start and end coordinates
    """
    Wm = np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2)

    img_edgemap = cv2.Canny(img, thresh_low, thresh_high)
    chamfer_dist = cv2.filter2D(img_edgemap.astype(float), -1, Wm)

    rect = Wm.shape
    index = np.unravel_index(np.argmax(chamfer_dist), img.shape[:-1])
    if printval:
        print(index)

    # 将x, y坐标转换为整数
    x, y = int(index[0]-rect[0]/2), int(index[1]-rect[1]/2)
    im = img.copy()
    # 修正了矩形对角线坐标点的计算方式
    cv2.rectangle(im, (y, x), (y + rect[1], x + rect[0]), (255, 0, 0), 3)  # 加粗以更清晰显示
    return im, (x, y), (rect[0], rect[1])