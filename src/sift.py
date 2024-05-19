import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import patches as patches
import os
import math
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

# 建立SIFT特徵擷取器
sift = cv2.SIFT_create()

# 建立FLANN匹配對象（Fast Library for Approximate Nearest Neighbors）
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
searchParams = dict(checks=100)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# 根據與最近點距離決定是否匹配
# 程式會先計算與第一接近點的距離，再計算與第二接近點的距離，如果與第二接近點的距離小於(與第一接近點的距離*ratio)，則判定兩者為匹配點，否則會被隱藏
def getMatchNum(matches, ratio):

    '''返回特徵點匹配數量和matchesMask'''
    
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            matchesMask[i] = [1, 0] #在遮罩中顯示
            matchNum += 1
    return (matchNum, matchesMask)

def convertOpponent(image, channel):
    
    '''SIFT本身對顏色很不敏感，沒辦法區分形狀相似但顏色不相似的物件，因此先將圖片轉換成Opponent Color Space(能用灰值描述色彩訊息)'''

    try:
        # Convert the image to float32
        image = np.float32(image) / 255.0

        # Split into color channels
        B, G, R = cv2.split(image)

        # Convert to Opponent Color Space
        O1 = (R - G) / np.sqrt(2)
        O2 = (R + G - 2 * B) / np.sqrt(6)
        O3 = (R + G + B) / np.sqrt(3)

        # Scale back to the range [0, 255]
        O1 = cv2.normalize(O1, None, 0, 255, cv2.NORM_MINMAX)
        O2 = cv2.normalize(O2, None, 0, 255, cv2.NORM_MINMAX)
        O3 = cv2.normalize(O3, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8
        O1 = np.uint8(O1)
        O2 = np.uint8(O2)
        O3 = np.uint8(O3)

        # Return the specified channel
        if channel == 'O1':
            return O1
        elif channel == 'O2':
            return O2
        elif channel == 'O3':
            return O3
        else:
            raise ValueError("Invalid channel selected. Choose from 'O1', 'O2', 'O3'.")
        
    except Exception as e:
        print(f"Error in convertOpponent: {e}")
        return None

def SIFT_main(sampleImage, queryImage, channel):

    '''
    和已知浮水印進行SIFT特徵匹配，並回傳匹配點
    '''

    comparisonImageList = []

    # 載入浮水印並提取特徵
    sampleImage = convertOpponent(sampleImage, channel=channel)
    #sampleImage = cv2.Canny(sampleImage, 100, 180)
    #sampleImage = cv2.equalizeHist(sampleImage)
    if sampleImage is None:
        print("Error: Failed to load sample image.")
        return comparisonImageList

    kp1, des1 = sift.detectAndCompute(sampleImage, None)  # 提取浮水印特徵
    originalImage = queryImage.copy()
    queryImage = convertOpponent(queryImage, channel=channel)
    #queryImage = cv2.Canny(queryImage, 50, 180)
    #queryImage = cv2.equalizeHist(queryImage)

    if queryImage is None:
        print(f"Error: Failed to load query image.")
        return comparisonImageList

    try:
        kp2, des2 = sift.detectAndCompute(queryImage, None) # 提取待比對圖像中的特徵

        # 若無法提取特徵，警告並跳過
        if des2 is None:
            print(f"Error: Failed to extract features from query image.")
            return comparisonImageList

        matches = flann.knnMatch(des1, des2, k=2) #進行特徵點兩兩匹配
        matchNum, matchesMask = getMatchNum(matches, 0.95) # 通過比率條件，計算出匹配程度
        matchRatio = matchNum * 100 / len(matches)

        drawParams = dict(matchColor=(0, 255, 0),
                            singlePointColor=(255, 0, 0),
                            matchesMask=matchesMask,
                            flags=0)
        
        # 將兩個圖像的特徵點匹配可視化
        comparisonImage = cv2.drawMatchesKnn(sampleImage, kp1, queryImage, kp2, matches, None, **drawParams)

        # 提取匹配點的位置
        matched_points_kp2 = [kp2[m[0].trainIdx].pt for m, mask in zip(matches, matchesMask) if mask[0] == 1]
        match_points = [(int(pt[0]), int(pt[1])) for pt in matched_points_kp2]


        comparisonImageList.append((comparisonImage, matchRatio, match_points, queryImage, originalImage)) # 紀錄結果

    # 若出現其他錯誤，提出警告
    except Exception as e:
        print(f"Error processing image: {e}")

    return comparisonImageList
  
def kmeans_outlier_removal(comparisonImageList, iter=1):

    """
    抓出來的匹配點仍有一些散落在真正的浮水印之外，假設匹配點大部分都會集中在浮水印上，可以使用Kmeans++刪掉稀疏點(群內距離較大)。
    """

    for _ in range(iter):
        for index, (image, ratio, match_points, queryImage, originalImage) in enumerate(comparisonImageList):

            match_points_array = np.array(match_points)

            # 先分一群
            num_clusters = 1  
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=200)
            kmeans.fit(match_points_array)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            distances = np.linalg.norm(match_points_array - cluster_centers[labels], axis=1)

            # 檢查群內距離
            distance_threshold = 100

            # 距離在容許範圍內，假定都在浮水印上
            if all(distance < distance_threshold for distance in distances):
                filtered_match_points = match_points_array.tolist()

            # 距離太大，則分兩群
            else:
                num_clusters = 2  
                kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=200)
                kmeans.fit(match_points_array)
                labels = kmeans.labels_

                cluster1_points = match_points_array[labels == 0].tolist()
                cluster2_points = match_points_array[labels == 1].tolist()

                avg_distance_cluster1 = np.mean(np.linalg.norm(cluster1_points - np.mean(cluster1_points, axis=0), axis=1))
                avg_distance_cluster2 = np.mean(np.linalg.norm(cluster2_points - np.mean(cluster2_points, axis=0), axis=1))

                # 找出較集中的那群並輸出
                if avg_distance_cluster1 <= avg_distance_cluster2:
                    filtered_match_points = cluster1_points
                else:
                    filtered_match_points = cluster2_points

            comparisonImageList[index] = (image, ratio, filtered_match_points, queryImage, originalImage)
    
    return comparisonImageList
  
def watermark_detector_SIFT(Wm_sample, img, gx, gy, thresh_low=200, thresh_high=220, iter=4, channel='O1', printval=False):

    """
    利用SIFT縮限偵測範圍
    """

    #匹配特徵點
    comparisonImageList = SIFT_main(Wm_sample, img, channel=channel) #channel 需微調
    # 縮限特徵點
    comparisonImageList = kmeans_outlier_removal(comparisonImageList, iter=iter) #iter 需微調

    padding = 20
    for image, ratio, match_points, queryImage, originalImage in comparisonImageList:

      # Extract the x and y coordinates from match_points
      x_coordinates = [point[0] for point in match_points]
      y_coordinates = [point[1] for point in match_points]

      # Calculate the bounding box corners with 30px padding
      min_x = int(min(x_coordinates)) - padding
      min_y = int(min(y_coordinates)) - padding
      max_x = int(max(x_coordinates)) + padding
      max_y = int(max(y_coordinates)) + padding

      # Ensure the bounding box is within the image boundaries
      min_x = max(0, min_x)
      min_y = max(0, min_y)
      max_x = min(queryImage.shape[1], max_x)
      max_y = min(queryImage.shape[0], max_y)

      # Crop the image based on the bounding box
      cropped_image = originalImage[min_y:max_y, min_x:max_x] # 根據偵測的區域裁減原圖

    Wm = np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2)

		# 用Canny再做chamfer distance
    img_edgemap = cv2.Canny(cropped_image, thresh_low, thresh_high)
    chamfer_dist = cv2.filter2D(img_edgemap.astype(float), -1, Wm, borderType=cv2.BORDER_CONSTANT)
    
    rect = Wm.shape
    index = np.unravel_index(np.argmax(chamfer_dist), cropped_image.shape[:-1])
    if printval:
        print(index)

    # 將x, y座標轉換為矩形在cropped_image的左上角座標
    x, y = int(index[0]-rect[0]/2), int(index[1]-rect[1]/2)
    # 將x, y座標轉換為矩形在原圖的左上角座標
    x, y = x+min_y, y+min_x
    # 將x, y座標轉換為矩形的右下角座標
    x_end, y_end = int(x + rect[0]), int(y + rect[1])
    
    # 畫出矩形
    im = img.copy()
    cv2.rectangle(im , (y, x), (y + rect[1], x + rect[0]), (255, 0, 0), 3) 

    #im[min_y:max_y, min_x:max_x] = cropped_image_copy
    
    return (im, (x, y), (x_end, y_end), cropped_image)