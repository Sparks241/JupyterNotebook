import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_and_preprocess_data(file_path):  
    """加载数据并进行预处理"""
    try:
        # 读取 CSV 文件
        print(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        print(f"数据基本信息：")
        df.info()
        
        # 查看数据集行数和列数
        rows, columns = df.shape
        
        if rows < 100:
            # 小数据集（行数少于100）查看全量数据信息
            print(f"数据全部内容信息：")
            print(df.to_csv(sep='\t', na_rep='nan'))
        else:
            # 大数据集查看数据前几行信息
            print(f"数据前几行内容信息：")
            print(df.head().to_csv(sep='\t', na_rep='nan'))
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n缺失值统计：")
            print(missing_values[missing_values > 0])
            # 处理缺失值（这里选择删除含有缺失值的行，可根据需要修改）
            df = df.dropna()
            print(f"删除缺失值后，数据形状: {df.shape}")
        
        # 查看数据集行数和列数
        rows, columns = df.shape
        
        # 检查重复值
        if df.duplicated().sum() > 0:
            print(f"\n发现{df.duplicated().sum()}条重复数据")
            # 删除重复数据
            df = df.drop_duplicates()
            print(f"删除重复值后，数据形状: {df.shape}")
        
        # 提取特征列（假设除了前两列都是特征）
        # 注意：根据实际数据调整特征列的选择
        feature_columns = df.columns[2:] if rows > 0 and columns > 2 else df.columns
        X = df[feature_columns]
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"数据预处理完成，特征列: {', '.join(feature_columns)}")
        return df, X_scaled, feature_columns, scaler
    
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请确保文件在正确的路径下。")
        return None, None, None, None
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return None, None, None, None

def find_optimal_clusters(X_scaled, max_clusters=10):
    """使用肘部法则和轮廓系数寻找最佳聚类数量"""
    if X_scaled is None:
        return 2  # 默认返回2个聚类
    
    print(f"\n正在寻找最佳聚类数量（1-{max_clusters}）...")
    wss = []  # 簇内误差平方和
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 计算簇内误差平方和
        wss.append(kmeans.inertia_)
        
        # 计算轮廓系数
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
        
        print(f"聚类数量: {k}, 轮廓系数: {score:.4f}")
    
    # 找出轮廓系数最高的聚类数量
    best_k = np.argmax(silhouette_scores) + 2
    
    # 绘制肘部法则图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), wss, 'bo-')
    plt.xlabel('聚类数量')
    plt.ylabel('簇内误差平方和 (WSS)')
    plt.title('肘部法则图')
    plt.grid(True)
    
    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'go-')
    plt.xlabel('聚类数量')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数图')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    print(f"已保存最佳聚类数量图: optimal_clusters.png")
    
    # 显示图表
    plt.show()
    
    print(f"\n根据轮廓系数，最佳聚类数量为: {best_k}")
    return best_k

def perform_clustering(X_scaled, n_clusters):
    """执行聚类分析"""
    if X_scaled is None:
        return None, None, None
    
    print(f"\n正在执行聚类分析 (k={n_clusters})...")
    
    # 使用K-means算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 使用层次聚类算法进行聚类
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    print(f"聚类分析完成")
    return kmeans, kmeans_labels, hierarchical_labels

def visualize_clusters(X_scaled, kmeans, kmeans_labels, hierarchical_labels, feature_columns):
    """可视化聚类结果"""
    if X_scaled is None or kmeans is None:
        return
    
    print("\n正在生成聚类可视化图表...")
    
    # 使用PCA降维以便可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 计算方差解释率
    explained_variance = pca.explained_variance_ratio_
    
    # 将K-means质心转换到PCA空间
    kmeans_centers_pca = pca.transform(kmeans.cluster_centers_)
    
    # 绘制K-means聚类结果
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans_centers_pca[:, 0], kmeans_centers_pca[:, 1], c='red', s=200, alpha=0.7, label='质心')
    plt.title('K-means聚类结果')
    plt.xlabel(f'主成分1 (解释方差: {explained_variance[0]:.2%})')
    plt.ylabel(f'主成分2 (解释方差: {explained_variance[1]:.2%})')
    plt.legend()
    plt.grid(True)
    
    # 添加颜色条
    plt.colorbar(scatter, label='聚类')
    
    # 绘制层次聚类结果
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title('层次聚类结果')
    plt.xlabel(f'主成分1 (解释方差: {explained_variance[0]:.2%})')
    plt.ylabel(f'主成分2 (解释方差: {explained_variance[1]:.2%})')
    plt.grid(True)
    
    # 添加颜色条
    plt.colorbar(scatter, label='聚类')
    
    plt.tight_layout()
    plt.savefig('clustering_results.png')
    print(f"已保存聚类结果图: clustering_results.png")
    
    # 显示图表
    plt.show()
    
    # 3D可视化
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(14, 6))
    
    # K-means 3D可视化
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                         c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
    kmeans_centers_3d = pca_3d.transform(kmeans.cluster_centers_)
    ax1.scatter(kmeans_centers_3d[:, 0], kmeans_centers_3d[:, 1], kmeans_centers_3d[:, 2],
               c='red', s=200, alpha=0.7, label='质心')
    ax1.set_title('K-means聚类结果(3D)')
    ax1.set_xlabel(f'主成分1')
    ax1.set_ylabel(f'主成分2')
    ax1.set_zlabel(f'主成分3')
    plt.legend()
    plt.colorbar(scatter, ax=ax1, label='聚类')
    
    # 层次聚类 3D可视化
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                         c=hierarchical_labels, cmap='viridis', s=50, alpha=0.7)
    ax2.set_title('层次聚类结果(3D)')
    ax2.set_xlabel(f'主成分1')
    ax2.set_ylabel(f'主成分2')
    ax2.set_zlabel(f'主成分3')
    plt.colorbar(scatter, ax=ax2, label='聚类')
    
    plt.tight_layout()
    plt.savefig('clustering_results_3d.png')
    print(f"已保存3D聚类结果图: clustering_results_3d.png")
    
    # 显示图表
    plt.show()

def analyze_cluster_features(df, X_scaled, kmeans, kmeans_labels, feature_columns, scaler, n_clusters):
    """分析聚类特征"""
    if df is None or X_scaled is None or kmeans is None:
        return
    
    print("\n正在分析聚类特征...")
    
    # 将聚类标签添加到原始数据
    df['Cluster'] = kmeans_labels
    
    # 计算每个聚类的样本数量
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print("\n各聚类样本数量:")
    print(cluster_counts)
    
    # 反标准化以便更好地理解特征
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_columns,
        index=[f'聚类 {i}' for i in range(n_clusters)]
    )
    
    print("\n各聚类的特征平均值:")
    print(cluster_centers)
    
    # 绘制雷达图比较聚类中心
    plt.figure(figsize=(12, 10))
    
    angles = np.linspace(0, 2*np.pi, len(feature_columns), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    for i in range(n_clusters):
        values = cluster_centers.iloc[i].values.tolist()
        values += values[:1]  # 闭合雷达图
        plt.polar(angles, values, 'o-', linewidth=2, label=f'聚类 {i} (n={cluster_counts[i]})')
    
    plt.thetagrids(np.degrees(angles[:-1]), feature_columns)
    plt.title('各聚类的特征比较')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.savefig('cluster_features_radar.png')
    print(f"已保存聚类特征雷达图: cluster_features_radar.png")
    
    # 显示图表
    plt.show()
    
    # 绘制热力图
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_centers, annot=True, cmap='viridis', fmt='.2f')
    plt.title('各聚类的特征热力图')
    plt.tight_layout()
    plt.savefig('cluster_features_heatmap.png')
    print(f"已保存聚类特征热力图: cluster_features_heatmap.png")
    
    # 显示图表
    plt.show()
    
    # 绘制箱线图比较各聚类特征
    print(f"\n正在生成各特征箱线图...")
    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df)
        plt.title(f'各聚类的{feature}分布')
        plt.tight_layout()
        plt.savefig(f'cluster_{feature}_boxplot.png')
    
    print(f"已保存所有特征箱线图")

def main():
    """主函数"""
    file_path = "D:\第三阶段作业\Wholesale customers data.csv"
    
    print("===== 批发客户数据聚类分析 =====")
    
    # 加载并预处理数据
    df, X_scaled, feature_columns, scaler = load_and_preprocess_data(file_path)
    
    if df is not None and X_scaled is not None:
        # 寻找最佳聚类数量
        best_k = find_optimal_clusters(X_scaled)
        
        # 执行聚类
        kmeans, kmeans_labels, hierarchical_labels = perform_clustering(X_scaled, best_k)
        
        # 可视化聚类结果
        visualize_clusters(X_scaled, kmeans, kmeans_labels, hierarchical_labels, feature_columns)
        
        # 分析聚类特征
        analyze_cluster_features(df, X_scaled, kmeans, kmeans_labels, feature_columns, scaler, best_k)
        
        # 保存聚类结果到CSV
        df.to_csv('wholesale_customers_clustered.csv', index=False)
        print(f"\n已保存聚类结果到: wholesale_customers_clustered.csv")
        
        print("\n===== 聚类分析全部完成 =====")

if __name__ == "__main__":
    main()    