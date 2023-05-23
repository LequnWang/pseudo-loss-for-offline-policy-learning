import numpy as np

def GetCost(A, L, P, beta):
    data_size = A.shape[0]
    num_classes = P.shape[1]
    C = np.zeros((data_size, num_classes))
    for i in range(data_size):
        for j in range(num_classes):
            if A[i] == j:
                C[i, j] += L[i] / P[i, j]
            C[i, j] += beta / P[i, j]
    return C

def IPW(A, A_hat, L, P):
    return np.mean(np.sum(A * A_hat, axis=1) * L / np.sum((P * A), axis=1))

def PesudoLoss(A_hat, P):
    return np.mean(1 / np.sum(A_hat * P, axis=1))

def SampleVariance(A, A_hat, L, P):
    sample = np.sum(A * A_hat, axis=1) * L / np.sum((P * A), axis=1)
    sample_mean = np.mean(sample)
    return np.sum((sample - sample_mean) ** 2) / (sample.shape[0] - 1)
