import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression

from sklearn import cluster

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import itertools

import parse_xvl as xvl
import color_maps as cmaps
import vectorize_xvl_data as xvlvec
from color_palette import color_names



def train_learn_predict(xvl_vec_data, use_PCA=False):
    vectors = xvl_vec_data['vectors']
    labels = xvl_vec_data['labels']

    if use_PCA:
        pca = decomposition.PCA(n_components=5)
        pca.fit(vectors, labels)
        vectors = pca.transform(vectors)

    print(xvl_vec_data['labels_map'])
    print(vectors[2:5])

    tst_count = 50

    X = vectors[:-tst_count]
    Y = labels[:-tst_count]

    X_tst = vectors[-tst_count:]
    Y_tst = labels[-tst_count:]

    classifier = GaussianNB()
    # classifier = svm.SVC(decision_function_shape='ovo', kernel='linear', C=2.2)
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    # classifier = LinearDiscriminantAnalysis(solver='svd', store_covariance=True, n_components=100)
    # classifier = LogisticRegression()
    # classifier = DecisionTreeClassifier()
    # classifier = RandomForestClassifier(n_estimators=200,
    #                                     warm_start=True, oob_score=True,
    #                                     max_features=None,
    #                                     random_state=None)
    classifier.fit(X, Y)

    print(list(classifier.predict(X_tst)))
    print(Y_tst)
    print(classifier.fit(X, Y).score(X_tst, Y_tst))

    X_pca = show_pca_transform(X, Y)
    show_2D_projections(X_pca, Y)


def test_learn_predict(lrn_vec_data, tst_vec_data, use_PCA=False):
    labels_map = lrn_vec_data['labels_map']
    X = lrn_vec_data['vectors']
    Y = lrn_vec_data['labels']
    X_tst = tst_vec_data['vectors']

    if use_PCA:
        pca = decomposition.PCA(n_components=5)
        pca.fit(X, Y)
        X = pca.transform(X)
        X_tst = pca.transform(X_tst)

    # classifier = GaussianNB()
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    classifier = svm.SVC(decision_function_shape='ovo', kernel='linear', C=2.21)
    # classifier = RandomForestClassifier(n_estimators=200,
    #                                     warm_start=True, oob_score=True,
    #                                     max_features=None,
    #                                     random_state=None)
    classifier.fit(X, Y)
    predicted_labels = list(classifier.predict(X_tst))
    # labels = [labels_map[str(label)] for label in predicted_labels] # JSON keys has str-type
    labels = [labels_map[label] for label in predicted_labels]
    return labels


def mean_rgb_labels(xvl_data):
    rgb_map = ('r', 'g', 'b')
    hex_vectors = xvl.vectors_of_xvl_data(xvl_data)
    rgb_vectors = [[cmaps.hex2rgb(c) for c in v] for v in hex_vectors]
    rgb_mean = xvlvec.mean_by_indices(rgb_vectors, list(range(9)))
    labels = [rgb_map[np.argmax(rgb)] for rgb in rgb_mean]
    return labels


def show_2D_projections(X, Y):
    surfaces = list(itertools.combinations(list(range(len(X[0]))), 2))

    X_a = np.array(X)

    print(len(surfaces), len(X[0]))

    for pairidx, pair in enumerate(surfaces):
        plt.subplot(1, 3, pairidx + 1)
        plt.xlabel(str(pair[0]))
        plt.ylabel(str(pair[1]))
        plt.scatter(X_a[:, pair[0]], X_a[:, pair[1]], c=Y)
    plt.show()


def show_pca_transform(X, Y):
    pca = decomposition.PCA(n_components=3)
    pca.fit(X, Y)
    X_pca = pca.transform(X)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=Y)
    plt.show()
    return X_pca


def vectorize_xvl_color_file(xvl_file, json_file='', palette=[]):
    xvl_data = xvl.parse_xvl_color_matrix_file(xvl_file)

    # if not palette:
    #     color_list = list(xvl.xvl_data_color_set(xvl_data))
    #     _, palette = cmaps.make_cluster_map(color_list, n_clusters=150)

    xvl_vec_data = xvlvec.make_xvl_color_vec_data(xvl_data, palette)

    if json_file:
        xvl.save_to_json(xvl_vec_data, json_file)

    return xvl_vec_data


def compare_labels(labels_a, labels_b):
    if len(labels_a) == len(labels_b):
        diff = [i for i in range(len(labels_a)) if labels_a[i] != labels_b[i]]
        print(1.0 - len(diff)/len(labels_a))
        print(diff)


def train():
    palette = list(color_names.values())
    lrn_vec_data = vectorize_xvl_color_file("rgb_mean.xvl", palette=[])
    # lrn_vec_data = xvl.load_from_json("rgb_vec.json")
    train_learn_predict(lrn_vec_data, use_PCA=False)


def test():
    palette = list(color_names.values())
    lrn_vec_data = vectorize_xvl_color_file("rgb.xvl", palette=[])
    tst_vec_data = vectorize_xvl_color_file("rgb_tst.xvl", palette=[]) # palette=lrn_vec_data['palette'])

    my_xvl_data = xvl.parse_xvl_color_matrix_file("rgb_my.xvl")
    my_labels = xvl.labels_of_xvl_data(my_xvl_data)
    print(my_labels)

    # lrn_vec_data = xvl.load_from_json("rgb_vec.json")
    # tst_vec_data = xvl.load_from_json("rgb_tst_vec.json")

    res_labels = test_learn_predict(lrn_vec_data, tst_vec_data, use_PCA=False)

    print("predicted:")
    print(res_labels)
    compare_labels(my_labels, res_labels)
    xvl.set_labels_to_xvl_file("rgb_tst.xvl", "rgb_tst_res.xvl", res_labels)

    tst_xvl_data = xvl.parse_xvl_color_matrix_file("rgb_tst.xvl")
    mean_labels = mean_rgb_labels(tst_xvl_data)
    print("mean:")
    print(mean_labels)
    compare_labels(my_labels, mean_labels)
    xvl.set_labels_to_xvl_file("rgb_tst.xvl", "rgb_tst_mean.xvl", mean_labels)


def cluster_color_matrices():
    xvl_file = "rgb.xvl"
    xvl_data = xvl.parse_xvl_color_matrix_file(xvl_file)
    xvl_vec_data, _ = xvlvec.vectorize_xvl_color_data(xvl_data, palette=[])
    vectors = [v for l, v in xvl_vec_data]
    print(vectors)
    # xvl_vec_data = vectorize_xvl_color_file(xvl_file, palette=[])
    # vectors = xvl_vec_data['vectors']

    kmeans_model = cluster.KMeans(n_clusters=10)
    idx = kmeans_model.fit_predict(vectors)

    labels = [str(i) for i in idx]
    xvl.set_labels_to_xvl_file(xvl_file, "rgb_cluster.xvl", labels)

# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def cluster_figures():
    xvl_file = "sense.xvl"
    xvl_data = xvl.parse_xvl_figures_file(xvl_file)
    xvl_vec_data = xvlvec.make_xvl_figures_vec_data(xvl_data)

    vectors = np.array(xvl_vec_data['vectors'])
    print(len(vectors), vectors[:3])

    n_clusters = 80

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(vectors, quantile=0.3)

    cluster_model = cluster.KMeans(n_clusters=n_clusters)
    # cluster_model = cluster.MeanShift()
    # cluster_model = cluster.SpectralClustering(n_clusters=n_clusters,
    #                                           eigen_solver='arpack',
    #                                           affinity="nearest_neighbors")
    # cluster_model =  cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    idx = cluster_model.fit_predict(vectors)

    labels = [str(i) for i in idx]
    xvl.set_labels_to_xvl_file(xvl_file, "sense_cluster.xvl", labels)


if __name__ ==  "__main__":
    # train()
    # test()
    # cluster_color_matrices()
    cluster_figures()

    # rgb_xvl_data = xvl.parse_xvl_color_matrix_file("rgb_my.xvl")
    # mean_labels = mean_rgb_labels(rgb_xvl_data)
    # xvl.set_labels_to_xvl_color_matrix_file("rgb_my.xvl", "rgb_my.xvl", mean_labels)
    # compare_labels(xvl.labels_of_xvl_data(rgb_xvl_data), mean_labels)
