'''
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : fundamental_mat.py
#
# Written by Dongwon Kim
#
# Fundamental Matrix
#   finding Fundamental matrix, epipolar line
#
# Modificatoin history
#   written by Dongwon Kim on Dec 03, 2021
#
# OpenCV version: 4.5.3
# using virtual environment with Anaconda
# 
# notice
#   tried RANSAC for compute_F_mine, but findling local minimum gives better result in average
'''
import numpy as np
import time
import cv2
from data.compute_avg_reproj_error import *

SEED = 0
INF = float('inf')
np.random.seed(SEED)

# file path
TEMPLE1_PATH = './data/temple1.png'
TEMPLE2_PATH = './data/temple2.png'
HOUSE1_PATH = './data/house1.jpg'
HOUSE2_PATH = './data/house2.jpg'
LIBRARY1_PATH = './data/library1.jpg'
LIBRARY2_PATH = './data/library2.jpg'
TEMPLEM_PATH = './data/temple_matches.txt'
HOUSEM_PATH = './data/house_matches.txt'
LIBRARYM_PATH = './data/library_matches.txt'

# open matching information text
templeM = np.loadtxt(TEMPLEM_PATH)
houseM = np.loadtxt(HOUSEM_PATH)
libraryM = np.loadtxt(LIBRARYM_PATH)

# [x, y] pairs from txt
def make_pair(M):
    src_point = []
    dst_point = []
    for i in range(len(M)):
        src_point.append((M[i][0], M[i][1]))
        dst_point.append((M[i][2], M[i][3]))
    src_point = np.array(src_point, dtype=np.int32)
    dst_point = np.array(dst_point, dtype=np.int32)
    return src_point, dst_point

"""
randomly select n matches from M

return [x, y] pairs for src and dst shape: (n, 2)
""" 
def rand_select(M, n):
    src_point, dst_point = make_pair(M)
    corr = np.random.choice(src_point.shape[0], n)
    src = src_point[corr]
    dst = dst_point[corr]
    return src, dst

def compute_F_raw(M):    
    src, dst = rand_select(M, 8)
    A = []
    for i in range(8):
        x1 = src[i][0]
        y1 = src[i][1]
        x2 = dst[i][0]
        y2 = dst[i][1]
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    u, s, v = np.linalg.svd(A)
    f = np.reshape(v[8], (3, 3)) # smallest singular value locates at last(8) in s  

    return f

"""
make transform matrix for normalization
    zero centric, unit vector
return 3 x 3 matrix for src and dst
"""
def get_transform_mat(src, dst):
    avg_x1 = np.average(src[:, 0])
    avg_y1 = np.average(src[:, 1])
    avg_x2 = np.average(dst[:, 0])
    avg_y2 = np.average(dst[:, 1])
    
    cen_x1 = src[:, 0] - avg_x1
    cen_y1 = src[:, 1] - avg_y1
    cen_x2 = dst[:, 0] - avg_x2
    cen_y2 = dst[:, 1] - avg_y2
    
    std_x1 = np.linalg.norm(cen_x1)
    std_y1 = np.linalg.norm(cen_y1)
    std_x2 = np.linalg.norm(cen_x2)
    std_y2 = np.linalg.norm(cen_y2)
    
    T1 = np.array([[1/std_x1, 0, -avg_x1/std_x1], [0, 1/std_y1, -avg_y1/std_y1], [0, 0, 1]], dtype=np.float32)
    T2 = np.array([[1/std_x2, 0, -avg_x2/std_x2], [0, 1/std_y2, -avg_y2/std_y2], [0, 0, 1]], dtype=np.float32)
    return T1, T2

def compute_F_norm(M):
    src, dst = rand_select(M, 8)
    A = []
        
    T1, T2 = get_transform_mat(src, dst)
    
    X1 = np.c_[src[:, :], np.ones((8, 1))].transpose()
    X2 = np.c_[dst[:, :], np.ones((8, 1))].transpose()
    normX1 = np.matmul(T1, X1)
    normX2 = np.matmul(T2, X2)
   
    for i in range(8):
        x1 = normX1[0][i]
        y1 = normX1[1][i]
        x2 = normX2[0][i]
        y2 = normX2[1][i]
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])  
    
    _, _, v = np.linalg.svd(A)
    f = np.reshape(v[8], (3, 3))
    
    # compute SVD of fundamental matrix for limit rank 2
    u, s, V = np.linalg.svd(f)
    s[-1] = 0   
    S = np.zeros((u.shape[1], V.shape[0]))
    for i in range(len(s)):
        S[i, i] = s[i]
        
    normF= np.matmul(np.matmul(u, S), V)
    
    # back to original scale
    T2 = T2.transpose()
    finalF = np.matmul(np.matmul(T2, normF), T1)
    
    return finalF

"""
compute distance
sqrt((x1 - x2)**2 + (y1 - y2)**2) for each point

return sum of distance shape: (length of A, )
"""
def distance2(A, B):
    x_distance = (A[0, :] - B[0, :])**2
    y_distance = (A[1, :] - B[1, :])**2
    dis = x_distance + y_distance
    dis = np.sqrt(dis)
    
    return dis

"""
return fundamental matrix that has local minimum distance
inputs: 
    M: matches
    N: number of points to be used finding fundamental mat
        should be greater than 8 to solve equation
1. randomly select N number of points from matches
2. compute normalized fundamental mat using N matches
3. compute distance(error) between estimated point and real point
4. find mat that has local minimum distance(min distance for whole iteration)

return: 3 x 3 normalized fundamental mat
"""
def compute_F_mine(M, N):
    bestF = np.zeros([3,3])
    bestT1 = np.zeros([3,3])
    bestT2 = np.zeros([3,3])
    total_iter = 3000
    min_distance = INF
    
    for _ in range(total_iter):
        src, dst = rand_select(M, N)
        A = []
        
        T1, T2 = get_transform_mat(src, dst)
        
        X1 = np.c_[src[:, :], np.ones((N, 1))].transpose()
        X2 = np.c_[dst[:, :], np.ones((N, 1))].transpose()
        normX1 = np.matmul(T1, X1)
        normX2 = np.matmul(T2, X2)
    
        for i in range(N):
            x1 = normX1[0][i]
            y1 = normX1[1][i]
            x2 = normX2[0][i]
            y2 = normX2[1][i]
            A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])  
        
        _, _, v = np.linalg.svd(A)
        f = np.reshape(v[-1], (3, 3))
        
        u, s, V = np.linalg.svd(f)
        s[-1] = 0   
        S = np.zeros((u.shape[1], V.shape[0]))
        for i in range(len(s)):
            S[i, i] = s[i]
            
        f = np.matmul(np.matmul(u, S), V)
        
        T2 = T2.transpose()
        f = np.matmul(np.matmul(T2, f), T1)

        ft = f.transpose()
        Fx1 = np.matmul(f, X1)
        Ftx2 = np.matmul(ft, X2)
        distance = distance2(X2, Fx1) + distance2(X1, Ftx2)
        total_dis = np.sum(distance)
        if(total_dis < min_distance):
            min_distance = total_dis
            bestF = f
            
    return bestF

# compute fundamental matrix and error
templeF_raw = compute_F_raw(templeM)
houseF_raw = compute_F_raw(houseM)
libraryF_raw = compute_F_raw(libraryM)
templeE_raw = compute_avg_reproj_error(templeM, templeF_raw)
houseE_raw = compute_avg_reproj_error(houseM, houseF_raw)
libraryE_raw = compute_avg_reproj_error(libraryM, libraryF_raw)

templeF_norm = compute_F_norm(templeM)
houseF_norm = compute_F_norm(houseM)
libraryF_norm = compute_F_norm(libraryM)
templeE_norm = compute_avg_reproj_error(templeM, templeF_norm)
houseE_norm = compute_avg_reproj_error(houseM, houseF_norm)
libraryE_norm = compute_avg_reproj_error(libraryM, libraryF_norm)

startTime = time.perf_counter()
templeF_mine = compute_F_mine(templeM, 16)
endTime = time.perf_counter()
print("[temple] computing time: {0:.5f}s".format(endTime - startTime))

startTime = time.perf_counter()
houseF_mine = compute_F_mine(houseM, 16)
endTime = time.perf_counter()
print("[house] computing time: {0:.5f}s".format(endTime - startTime))

startTime = time.perf_counter()
libraryF_mine = compute_F_mine(libraryM, 16)
endTime = time.perf_counter()
print("[library] computing time: {0:.5f}s".format(endTime - startTime))

templeE_mine = compute_avg_reproj_error(templeM, templeF_mine)
houseE_mine = compute_avg_reproj_error(houseM, houseF_mine)
libraryE_mine = compute_avg_reproj_error(libraryM, libraryF_mine)

# print the result
print("Average Reprojection Errors (temple1.png and temple2.png)")
print("\tRaw = {}".format(templeE_raw))
print("\tNorm = {}".format(templeE_norm))
print("\tMine = {}".format(templeE_mine))

print("Average Reprojection Errors (house1.png and house2.png)")
print("\tRaw = {}".format(houseE_raw))
print("\tNorm = {}".format(houseE_norm))
print("\tMine = {}".format(houseE_mine))

print("Average Reprojection Errors (library1.png and library2.png)")
print("\tRaw = {}".format(libraryE_raw))
print("\tNorm = {}".format(libraryE_norm))
print("\tMine = {}".format(libraryE_mine))


"""
# Visualization
"""
"""
compute epipolar line
l, m : matrix [a, b, c] where line = ax + by c
return 3 x 3 l, m (each has 3 rows, each row is [a, b, c])
"""
def get_epipolar(src, dst, F):
    l = []
    m = []
    src_vec = np.c_[src[:, :], np.ones((3, 1))].transpose()
    dst_vec = np.c_[dst[:, :], np.ones((3, 1))].transpose()
    Ft = F.transpose()
    for i in range(3):
        l.append(np.matmul(Ft, dst_vec[:, i]))
        m.append(np.matmul(F, src_vec[:, i]))
    l = np.array(l)
    m = np.array(m)
    return l, m

"""
draw circle and line for the image
"""
def drawing(src, dst, l, m, image1, image2):
    img1 = image1.copy()
    img2 = image2.copy()
    width1 = img1.shape[1]
    width2 = img2.shape[1]
    for i in range(3):
        x1 = 0
        y1 = int(-m[i, 2]/m[i, 1])
        x2 = width1
        y2 = int(-(m[i, 2] + m[i, 0]*width1)/m[i, 1])
        if(i == 0):
            img1 = cv2.circle(img1, (src[i, 0], src[i, 1]), 5, (0, 0, 255), -1)
            img1 = cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 1)
        elif(i == 1):
            img1 = cv2.circle(img1, (src[i, 0], src[i, 1]), 5, (0, 255, 0), -1)
            img1 = cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            img1 = cv2.circle(img1, (src[i, 0], src[i, 1]), 5, (255, 0, 0), -1)
            img1 = cv2.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
    for i in range(3):
        x1 = 0
        y1 = int(-l[i, 2]/l[i, 1])
        x2 = width2
        y2 = int(-(l[i, 2] + l[i, 0]*width2)/l[i, 1])
        if(i == 0):
            img2 = cv2.circle(img2, (dst[i, 0], dst[i, 1]), 5, (0, 0, 255), -1)
            img2 = cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 1)
        elif(i == 1):
            img2 = cv2.circle(img2, (dst[i, 0], dst[i, 1]), 5, (0, 255, 0), -1)
            img2 = cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            img2 = cv2.circle(img2, (dst[i, 0], dst[i, 1]), 5, (255, 0, 0), -1)
            img2 = cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    
    img = cv2.hconcat([img1, img2]) # show two images at once
    
    return img

temple1 = cv2.imread(TEMPLE1_PATH, cv2.IMREAD_COLOR)
temple2 = cv2.imread(TEMPLE2_PATH, cv2.IMREAD_COLOR)
house1 = cv2.imread(HOUSE1_PATH, cv2.IMREAD_COLOR)
house2 = cv2.imread(HOUSE2_PATH, cv2.IMREAD_COLOR)
library1 = cv2.imread(LIBRARY1_PATH, cv2.IMREAD_COLOR)
library2 = cv2.imread(LIBRARY2_PATH, cv2.IMREAD_COLOR)

while 1:
    key = cv2.waitKey(0)
    # quit
    if key == ord('q'):
        break
    # compute F again and show corresponding epipolar lines
    else:
        temple_src, temple_dst = rand_select(templeM, 3)
        house_src, house_dst = rand_select(houseM, 3)
        library_src, library_dst = rand_select(libraryM, 3)
        
        startTime = time.perf_counter()
        templeF = compute_F_mine(templeM, 16)
        endTime = time.perf_counter()
        print("[temple] computing time: {0:.5f}s".format(endTime - startTime))

        startTime = time.perf_counter()
        houseF = compute_F_mine(houseM, 16)
        endTime = time.perf_counter()        
        print("[house] computing time: {0:.5f}s".format(endTime - startTime))
        
        startTime = time.perf_counter()
        libraryF = compute_F_mine(libraryM, 16)
        endTime = time.perf_counter()
        print("[library] computing time: {0:.5f}s".format(endTime - startTime))
        
        l, m = get_epipolar(temple_src, temple_dst, templeF)
        img = drawing(temple_src, temple_dst, l, m, temple1, temple2)
        cv2.imshow("[temple]Press q to exit", img)
        l, m = get_epipolar(house_src, house_dst, houseF)
        img = drawing(house_src, house_dst, l, m, house1, house2)
        cv2.imshow("[house]Press q to exit", img)
        l, m = get_epipolar(library_src, library_dst, libraryF)
        img = drawing(library_src, library_dst, l, m, library1, library2)
        cv2.imshow("[library]Press q to exit", img)
cv2.destroyAllWindows()



