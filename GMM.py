# @author panzhenfu 2017/6/12
# @email panzhenfu20@126.com

#coding=utf-8
import numpy as np
import copy
import matplotlib.pyplot as plt
# 单个高斯值
def Gaussian_DN(X,U_Mean,Cov):
    D = np.shape(X)[0]
    Y = X-U_Mean
    temp = Y.T * (Cov+np.eye(D)*0.01).I * Y
    result = (1.0/((2*np.pi)**(D/2)))*(1.0/(np.linalg.det(Cov+np.eye(D)*0.01)**0.5))*np.longfloat(np.exp(-0.5*temp))
    return result

# 计算样本均值
def CalMean(X):
    D,N=np.shape(X)
    MeanVector=np.mat(np.zeros((D,1)))
    for d in range(D):
        for n in range(N):
            MeanVector[d,0] += X[d,n]
        MeanVector[d,0] /= float(N)
    return MeanVector

# 计算似然函数值
def maxLikelyhood(Xn, Pik, Uk, Cov):
    D, N = np.shape(Xn)
    D_k, K = np.shape(Uk)
    if D != D_k:
        print ('dimension not equal, break')
        return
    probility_mat = np.zeros((N, K))
    Likelyhood = 0.0
    for n_iter in range(N):
        temp = 0
        for k_iter in range(K):
            gaussian = Gaussian_DN(Xn[:, n_iter], Uk[:, k_iter], Cov[k_iter])
            probility_mat[n_iter, k_iter] = gaussian
            temp += Pik[0, k_iter] * gaussian
        Likelyhood += np.log(temp)
    return float(Likelyhood), probility_mat

def calgmm(Xn, Pik, Uk, Cov):
    D_k, K = np.shape(Uk)
    temp = 0;
    for k_iter in range(K):
        temp += Pik[0, k_iter]*Gaussian_DN(Xn, Uk[:, k_iter], Cov[k_iter])
    return temp

def EMforMixGaussian(InputData, K, MaxIter):
    # 初始化piK
    pi_Cof = np.mat(np.ones((1, K)) * (1.0 / float(K)))
    X = np.mat(InputData)
    X_mean = CalMean(X)
    print (X_mean)

    # 初始化uK，其中第k列表示第k个高斯函数的均值向量
    # X为D维，N个样本点
    D, N = np.shape(X)
    print (D, N)
    UK = np.mat(np.zeros((D, K)))
    for d_iter in range(D):
        for k_iter in range(K):
            UK[d_iter, k_iter] = X_mean[d_iter, 0] + np.random.uniform(-30, 30)
    print (UK)
    # 初始化k个协方差矩阵的列表
    List_cov = []

    for k_iter in range(K):
        List_cov.append(np.mat(np.eye(X[:, 0].size)))
    print (List_cov)

    Likelyhood_new = 0
    Likelyhood_old, _ = maxLikelyhood(X, pi_Cof, UK, List_cov)
    print (Likelyhood_old)
    currentIter = 0
    while True:
        currentIter += 1

        rZnk = np.mat(np.zeros((N, K)))
        denominator = np.mat(np.zeros((N, 1)))
        # rZnk=pi_k*Gaussian(Xn|uk,Cov_k)/sum(pi_j*Gaussian(Xn|uj,Cov_j))
        for n_iter in range(N):
            for k_iter in range(K):
                rZnk[n_iter, k_iter] = pi_Cof[0, k_iter] * Gaussian_DN(X[:, n_iter], UK[:, k_iter], List_cov[k_iter])
                denominator[n_iter, 0] += rZnk[n_iter, k_iter]
            for k_iter in range(K):
                rZnk[n_iter, k_iter] /= denominator[n_iter, 0]
                # print 'rZnk', rZnk[n_iter,k_iter]

        # Nk=sum(rZnk)
        Nk = np.mat(np.zeros((1, K)))
        pi_new = np.mat(np.zeros((1, K)))
        for k_iter in range(K):
            for n_iter in range(N):
                Nk[0, k_iter] += rZnk[n_iter, k_iter]
            pi_new[0, k_iter] = Nk[0, k_iter] / (float(N))
            # print 'pi_k_new',pi_new[0,k_iter]

        # uk_new= (1/sum(rZnk))*sum(rZnk*Xn)
        UK_new = np.mat(np.zeros((D, K)))
        for k_iter in range(K):
            for n_iter in range(N):
                UK_new[:, k_iter] += (1.0 / float(Nk[0, k_iter])) * rZnk[n_iter, k_iter] * X[:, n_iter]
                # print 'UK_new',UK_new[:,k_iter]

        List_cov_new = []
        for k_iter in range(K):
            X_cov_new = np.mat(np.zeros((D, D)))
            for n_iter in range(N):
                Temp = X[:, n_iter] - UK_new[:, k_iter]
                X_cov_new += (1.0 / float(Nk[0, k_iter])) * rZnk[n_iter, k_iter] * Temp * Temp.transpose()
                # print 'X_cov_new',X_cov_new
            List_cov_new.append(X_cov_new)

        Likelyhood_new, _ = maxLikelyhood(X, pi_new, UK_new, List_cov_new)
        print ('Likelyhood_new', Likelyhood_new, currentIter)

        if Likelyhood_old >= Likelyhood_new or currentIter > MaxIter:
            break
        UK = copy.deepcopy(UK_new)
        pi_Cof = copy.deepcopy(pi_new)
        List_cov = copy.deepcopy(List_cov_new)
        Likelyhood_old = Likelyhood_new
    return pi_Cof, UK, List_cov

# 生成随机数据，4个高斯模型
def generate_data(sigma, N, mu1, mu2, mu3, mu4, alpha):

    X = np.zeros((N, 2))  # 初始化X，2行N列。2维数据，N个样本
    X = np.matrix(X)

    for i in range(N):
        if np.random.random(1) < alpha[0]:  # 生成0-1之间随机数
            X[i, :] = np.random.multivariate_normal(mu1, sigma[0], 1)  # 用第一个高斯模型生成2维数据
        elif alpha[0] <= np.random.random(1) < alpha[0]+alpha[1]:
            X[i, :] = np.random.multivariate_normal(mu2, sigma[1], 1)  # 用第二个高斯模型生成2维数据
        elif alpha[0]+alpha[1]<= np.random.random(1) < alpha[0]+alpha[1]+alpha[2]:
            X[i, :] = np.random.multivariate_normal(mu3, sigma[2], 1)  # 用第三个高斯模型生成2维数据
        else:
            X[i, :] = np.random.multivariate_normal(mu4, sigma[3], 1)  # 用第四个高斯模型生成2维数据
    return X


if __name__ == "__main__":
    N = 500  # 样本数目
    k = 4  # 高斯模型数
    probility = np.zeros(N)  # 混合高斯分布
    u1 = [5, 35]
    u2 = [15, 60]
    u3 = [30, 50]
    u4 = [40, 20]
    sigma_list = []   # 协方差矩阵
    sigma_list.append(np.mat([[30, 0], [0, 10]]))
    sigma_list.append(np.mat([[25, 12], [12, 20]]))
    sigma_list.append(np.mat([[12, 5], [5, 15]]))
    sigma_list.append(np.mat([[45, 9], [9, 20]]))
    alpha = [0.1, 0.4, 0.3, 0.2]  # 混合项系数
    X = generate_data(sigma_list, N, u1, u2, u3, u4, alpha)  # 生成数据

    iter_num = 5000  # 迭代次数
    Pi, Uk, list_cov = EMforMixGaussian(np.mat(X).T, k, iter_num)

    print (Uk)
    print (list_cov)
    print (Pi)
      # 可视化结果
    # 画生成的原始数据
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c='g', s=25, alpha=0.4, marker='o')  # T散点颜色，s散点大小，alpha透明度，marker散点形状
    plt.title('random generated data')
    plt.scatter(Uk[0, :], Uk[1, :],c='r', s=40, alpha=1, marker='x')

    energy_new, excep = maxLikelyhood(np.mat(X).T, Pi, Uk, list_cov)
    print (excep)
    # 画分类好的数据
    plt.subplot(122)
    plt.title('classified data through EM')
    order = np.zeros(N)
    color = ['b', 'r', 'k', 'y']
    for i in range(N):
        for j in range(k):
            if excep[i, j] == max(excep[i, :]):
                order[i] = j  # 选出X[i,:]属于第几个高斯模型
        plt.scatter(X[i, 0], X[i, 1], c=color[int(order[i])], s=25, alpha=0.4, marker='o')  # 绘制分类后的散点图
    plt.show()


