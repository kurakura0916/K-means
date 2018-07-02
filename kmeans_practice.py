import numpy as np
import matplotlib.pyplot as plt

# ダミーデータを作成する関数
def generate_2dim_normal(mean, variance, covariance, sample_size):
    cov = [[variance,covariance],
           [covariance,variance]]
    return np.random.multivariate_normal(mean, cov, sample_size)

# 各データとシードの距離を計算し、最寄りのクラスターを返す関数
def get_cluster_id_by_euclidian_distance(data, center):

    #まずデータ点 x シードの数の全て0の行列を用意しておきます
    result = np.zeros([len(data), len(center)])

    #シードごとに、各データ点との距離を計算し、resultに格納します
    for i in range(len(center)):
        square_total = ((data - center[i])**2).sum(axis=1)
        euclidian_distance = np.sqrt(square_total)
        result[:, i] = euclidian_distance

    # argmin(axis=1) 行毎に最小のインデックスを返す
    #一番近いシート番号を取得します。
    cluster_labels = result.argmin(axis=1)
    return cluster_labels

# シードの更新をする関数
def update_center(prev_center, cluster_labels,cluster_num):
    # shape要素数を返す。◯行△列を返す
    new_center = np.zeros(prev_center.shape)
    for i in range(cluster_num):
        new_center[i, :] = data[cluster_labels==i].mean(axis=0)
    return new_center

# 重心の位置が更新されなくなるまで、繰り返す関数
def Kmeans(data, cluster_num, max_iter = 100):
    seed_idx = np.random.randint(len(data), size=cluster_num)
    init_center = data[seed_idx]

    for i in range(max_iter):
        cluster_labels =get_cluster_id_by_euclidian_distance(data, init_center)
        new_center =  update_center(prev_center=init_center, cluster_labels=cluster_labels,cluster_num=cluster_num)

        if  np.sqrt(np.sum((new_center - init_center)**2)) < 0.00001:
            break

        init_center = new_center

    return cluster_labels,new_center

# 可視化するための関数
def cluster_visualize(data, cluster_labels):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    cluster_ids = list(set(cluster_labels))
    for k in range(len(cluster_ids)):
        cluster_id = cluster_ids[k]
        label_ = "clutser = %d"%cluster_id
        data_by_cluster = data[cluster_labels == cluster_id]
        ax.scatter(data_by_cluster[:,0], data_by_cluster[:,1], c=colorlist[k], label = label_)

    ax.scatter(new_center[:,0],new_center[:,1],marker="x",color="black",label="center of gravity")
    ax.set_title(u"Clustering")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    # クラスター数の定義
    cluster_num = 4
    
    # ダミーデータの作成
    cluster1 = generate_2dim_normal(mean = [0, 8], variance=1, covariance=0, sample_size=500)
    cluster2 = generate_2dim_normal(mean = [-1, 0], variance=1, covariance=0, sample_size=500)
    cluster3 = generate_2dim_normal(mean = [10, 10], variance=1, covariance=0, sample_size=300)
    cluster4 = generate_2dim_normal(mean = [5, 5.5], variance=0.8, covariance=-0.1, sample_size=200)
    data = np.vstack((cluster1, cluster2, cluster3, cluster4))

    # dataからランダムにシードを決めるため、indexをrandintを使い生成
    seed_idx = np.random.randint(len(data), size=cluster_num)
    init_center = data[seed_idx]

    # クラスターの配列と重心の位置配列を返す
    updated_cluster_labels,new_center = Kmeans(data, cluster_num, max_iter = 10)
    print("クラスターラベル\n{}".format(updated_cluster_labels))
    print("重心の位置\n{}".format(new_center))

    # 色分けされたクラスターと重心の位置を描画したグラフを生成
    cluster_visualize(data, updated_cluster_labels)
