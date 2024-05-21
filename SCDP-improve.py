import numpy as np
import pandas as pd
import os
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from load import load_data

# 参数设置
epsilon_ = 5  # 差分隐私隐私预算
add_dp = 0  # 是否添加差分隐私
choice = 3


# 计算最佳sigma参数矩阵
def adaptive_sigma(dist_matrix):
    n = dist_matrix.shape[0]
    sigma_matrix = np.zeros((n, n))
    index_order = np.zeros((n, n))
    for i in range(n):
        sorted_indices = np.argsort(dist_matrix[i, :])
        # 构建一个字典，键为排序后的索引，值为原始索引位置
        index_mapping = {sorted_indices[i]: i for i in range(len(sorted_indices))}
        # 使用字典将排序后的索引映射回原始索引位置
        original_positions = [index_mapping[j] for j in range(len(sorted_indices))]
        index_order[i] = original_positions
        m = np.mean(dist_matrix[i, :])
        k = int(n * 0.25)        # k为最近邻个数
        distances = dist_matrix[i, :]        # 取出第i行的距离
        knn_distances = np.sort(distances)[:k]        # 排序并选取前k个最近邻距离
        sigma_i = np.mean(knn_distances)
        for j in range(n):
            sigma_matrix[i, j] = sigma_i * 0.5 + m * 0.5
            # if (index_order[i][j] + index_order[j][i]) != 0:
            #     sigma_matrix[i, j] = m / ((index_order[i][j] + index_order[j][i]) / 2)  # 计算自适应sigma矩阵
            # else:      # 处理分母为零的情况，赋予一个特定的值，采用均值
            #     sigma_matrix[i, j] = m
    return sigma_matrix


# 计算高斯核函数的相似性矩阵
def compute_similarity_matrix(x, sigma_matrix):
    pairwise_dist = pairwise_distances(x)
    n = pairwise_dist.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = np.exp(-pairwise_dist[i, j] ** 2 / (2 * sigma_matrix[i, j] ** 2))
    symmetrical_similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    return symmetrical_similarity_matrix


# 添加拉普拉斯噪声实现差分隐私
def add_laplace_noise(similarity_matrix, epsilon, sensitivity=1.0):
    noise = np.random.laplace(0, sensitivity / epsilon, similarity_matrix.shape)
    noisy_similarity = similarity_matrix + noise
    noisy_similarity = (noisy_similarity + noisy_similarity.T) / 2
    np.fill_diagonal(noisy_similarity, 0)  # 对角线元素设置为0
    return np.maximum(noisy_similarity, 0)  # 确保相似性矩阵非负


# 谱聚类算法
def spectral_clustering_with_dp(x, n_clusters, epsilon):
    pairwise_dist = pairwise_distances(x)
    sigma_matrix = adaptive_sigma(pairwise_dist)
    similarity_matrix = compute_similarity_matrix(x, sigma_matrix)
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
    labels = spectral_clustering_with_dp(x, n_clusters=k, epsilon=epsilon_)
    evaluate_clustering(x, labels, 'spectral_clustering.csv', y, data_name)
    visualize_clustering(x, labels)


if __name__ == "__main__":
    main()
