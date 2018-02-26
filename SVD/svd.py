# Copyright (c) 2018, Ziping Chen.
# All rights reserved.
#
# File: pca.py
# Github: techping

import cv2 as cv
import numpy as np

def svd_compress(data_mat, k):
    u, sigma, vt = np.linalg.svd(data_mat)
    sigma_k = np.mat(np.eye(k) * sigma[:k])
    recon_mat = u[:,:k] * sigma_k * vt[:k,:]
    return u, sigma, vt, recon_mat

img = cv.imread('../img/lenna.jpg', 0)
u, s, vt, r = svd_compress(img, 10)
new_img = np.asarray(r).astype(np.uint8)
cv.imshow('original_img', img)
cv.imshow('new_img', new_img)
cv.waitKey(0)