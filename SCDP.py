import numpy as np
import pandas as pd
import os
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from load import load_data

# 参数设置
sigma_ = 1  # 高斯核函数宽度
epsilon_ = 5  # 差分隐私隐私预算
add_dp = 0  # 是否添加差分隐私
choice = 1


# 计算高斯核函数的相似性矩阵
def compute_similarity_matrix(x, sigma):
    pairwise_dist = pairwise_distances(x)
    return np.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))


# 添加拉普拉斯噪声实现差分隐私
def add_laplace_noise(similarity_matrix, epsilon, sensitivity=1.0):
    noise = np.random.laplace(0, sensitivity / epsilon, similarity_matrix.shape)
    noisy_similarity = similarity_matrix + noise
    noisy_similarity = (noisy_similarity + noisy_similarity.T) / 2
    np.fill_diagonal(noisy_similarity, 0)  # 对角线元素设置为0
    return np.maximum(noisy_similarity, 0)  # 确保相似性矩阵非负


# 谱聚类算法
def spectral_clustering_with_dp(x, n_clusters, epsilon, sigma):
    similarity_matrix = compute_similarity_matrix(x, sigma)
    if add_dp == 1:
        similarity_matrix = add_laplace_noise(similarity_matrix, epsilon)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    clustering.fit(similarity_matrix)
    return clustering.labels_


# 评估聚类结果
def evaluate_clustering(x, labels, filename, y, data_name):
    silhouette = silhouette_score(x, labels)
    ch_score = calinski_harabasz_score(x, labels)
    rand_score = adjusted_rand_score(y, labels)  # 计算兰德系数
    # 创建数据帧
    data = {'Epsilon': [epsilon_],
            'Silhouette Score': [silhouette],
            'Calinski-Harabasz Score': [ch_score],
            'Rand Index': [rand_score],
            'add_dp': [add_dp],
            'data': [data_name]}
    df = pd.DataFrame(data)

    # 将数据帧写入CSV文件
    df.to_csv(filename, mode='a', index=False, header=not os.path.exists(filename))

    print("Evaluation results saved to", filename)
    print("Silhouette Score:", silhouette)
    print("Calinski-Harabasz Score:", ch_score)
    print("Rand Index:", rand_score)


# 可视化聚类结果
def visualize_clustering(x, labels):
    pca = PCA(n_components=2)
    pca.fit(x)
    X_r = pca.transform(x)
    plt.scatter(X_r[:, 0], X_r[:, 1], c=labels, cmap='rainbow', s=5)
    plt.title('Spectral Clustering with Differential Privacy')
    plt.show()


# 主函数
def main():
    x, y, data_name, k = load_data(choice)
    labels = spectral_clustering_with_dp(x, n_clusters=k, epsilon=epsilon_, sigma=sigma_)
    evaluate_clustering(x, labels, 'spectral_clustering.csv', y, data_name)
    visualize_clustering(x, labels)


if __name__ == "__main__":
    main()
