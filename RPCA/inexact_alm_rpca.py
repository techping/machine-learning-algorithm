'''
file: inexact_alm_rpca.py
author: Ziping Chen
github: techping

Copyright (c) 2018 Ziping Chen

This python code implements the inexact augmented Lagrange multiplier
method for Robust PCA.

Modified from: http://perception.csl.illinois.edu/matrix-rank/Files/inexact_alm_rpca.zip

% Below is the copyright information from inexact_alm_rpca.m:
%
% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

'''

import numpy as np
import math

def choosvd(n, d):
    if n <= 100:
        if d / n <= 0.02:
            y = 1
        else:
            y = 0
    elif n <= 200:
        if d / n <= 0.06:
            y = 1
        else:
            y = 0
    elif n <= 300:
        if d / n <= 0.26:
            y = 1
        else:
            y = 0
    elif n <= 400:
        if d / n <= 0.28:
            y = 1
        else:
            y = 0
    elif n <= 500:
        if d / n <= 0.34:
            y = 1
        else:
            y = 0
    else:
        if d / n <= 0.38:
            y = 1
        else:
            y = 0
    return y

def inexact_alm_rpca(D, lambda_ = -1, tol = 1e-7, maxIter = 1000):
    m, n = np.shape(D)
    if lambda_ < 0:
        lambda_ = 1 / math.sqrt(m)
    if tol < 0:
        tol = 1e-7
    if maxIter < 0:
        maxIter = 1000
    # initialize
    Y = D
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y.reshape((m * n, 1), order = 'F'), np.inf) / lambda_
    dual_norm = max(norm_two, norm_inf)
    Y = np.divide(Y, dual_norm)

    A_hat = np.zeros((m, n))
    E_hat = np.zeros((m, n))
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(D, 'fro')

    iter_ = 0
    total_svd = 0
    converged = False
    stopCriterion = 1
    sv = 10

    while not converged:
        iter_ = iter_ + 1
        temp_T = D - A_hat + (1 / mu) * Y
        E_hat = temp_T - lambda_ / mu
        E_hat[E_hat < 0] = 0
        E_hat_ = temp_T + lambda_ / mu
        E_hat_[E_hat_ > 0] = 0
        E_hat = E_hat + E_hat_
        del E_hat_
        U, S, VT = np.linalg.svd(D - E_hat + (1 / mu) * Y)
        diagS = S
        if choosvd(n, sv) == 1:
            S = np.eye(sv) * S[:sv]
            U = U[:, :sv]
            VT = VT[:sv, :]
        else:
            S = np.diag(S)
        svp = np.shape(diagS[diagS > 1 / mu])[0]
        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05 * n), n)

        A_hat = np.dot(np.dot(U[:, :svp], (np.eye(svp) * (diagS[:svp] - 1 / mu))), VT[:svp, :])
        total_svd = total_svd + 1
        Z = D - A_hat - E_hat
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        stopCriterion = np.linalg.norm(Z, 'fro') / d_norm
        if stopCriterion < tol:
            converged = True

        if total_svd % 10 == 0:
            print('#svd ' + str(total_svd) + ' r(A) ' + str(np.linalg.matrix_rank(A_hat))
                + ' |E|_0 ' + str(np.shape(E_hat[abs(E_hat) > 0])[0])
                + ' stopCriterion ' + str(stopCriterion))

        if not converged and iter_ > maxIter:
            print('Maximum iterations reached')
            converged = 1

        pass

    return A_hat, E_hat, iter_
