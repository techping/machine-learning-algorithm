# Copyright (c) 2018, Ziping Chen.
# All rights reserved.
#
# File: pca.py
# Github: techping

import cv2 as cv
import numpy as np

def pca_compress(data_mat, k = 9999999):
    mean_vals = np.mean(data_mat, axis = 0)
    mean_removed = data_mat - mean_vals
    cov_mat = np.cov(mean_removed, rowvar = 0)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_idx = np.argsort(eig_vals)
    eig_val_idx = eig_val_idx[:-(k + 1):-1]
    re_eig_vects = eig_vects[:,eig_val_idx]
    low_dim_data = mean_removed * re_eig_vects
    recon_mat = (low_dim_data * re_eig_vects.T) + mean_vals
    return mean_removed, low_dim_data, recon_mat

img = cv.imread('../img/lenna.jpg', 0)
m, l, r = pca_compress(img, 10)
new_img = np.asarray(r).astype(np.uint8)
cv.imshow('original_img', img)
cv.imshow('new_img', new_img)
cv.waitKey(0)