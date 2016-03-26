import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import itertools

import parse_xvl as xvl
import color_maps as cmaps
from vectorize_xvl_data import make_xvl_color_matrix_vec_data, mean_by_indices


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

    # classifier = svm.SVC(decision_function_shape='ovo', kernel='linear', C=0.3)
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=17, n_jobs=-1)
    # classifier = LinearDiscriminantAnalysis(solver='svd')
    classifier = GaussianNB()
    # classifier = DecisionTreeClassifier()
    # classifier = RandomForestClassifier(n_estimators=350,
    #                                     warm_start=True, oob_score=True,
    #                                     max_features=None,
    #                                     random_state=None)
    classifier.fit(X, Y)

    print(list(classifier.predict(X_tst)))
    print(Y_tst)
    print(classifier.fit(X, Y).score(X_tst, Y_tst))

    X_pca = show_pca_transform(X, Y)
    #show_2D_projections(X_pca, Y)


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


    classifier = GaussianNB()
    # classifier = RandomForestClassifier(n_estimators=350,
    #                                 warm_start=True, oob_score=True,
    #                                 max_features=None,
    #                                 random_state=None)
    classifier.fit(X, Y)
    predicted_labels = list(classifier.predict(X_tst))
    labels = [labels_map[str(label)] for label in predicted_labels]
    return labels


def mean_rgb_labels(xvl_data):
    rgb_map = ('r', 'g', 'b')
    hex_vectors = xvl.vectors_of_xvl_data(xvl_data)
    rgb_vectors = [[cmaps.hex2rgb(c) for c in v] for v in hex_vectors]
    rgb_mean = mean_by_indices(rgb_vectors, list(range(9)))
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


def vectorize_and_save_xvl_data(xvl_file, json_file, palette=[]):
    xvl_data = xvl.parse_xvl_color_matrix_file(xvl_file)

    if not palette:
        color_list = list(xvl.xvl_data_color_set(xvl_data))
        _, palette = cmaps.make_cluster_map(color_list, n_clusters=150)

    xvl_vec_data = make_xvl_color_matrix_vec_data(xvl_data, palette)
    xvl.save_to_json(xvl_vec_data, json_file)
    return xvl_vec_data


def train():
    lrn_vec_data = vectorize_and_save_xvl_data("rgb.xvl", "rgb_vec.json")
    lrn_vec_data = xvl.load_from_json("rgb_vec.json")
    train_learn_predict(lrn_vec_data, use_PCA=True)


def test():
    lrn_vec_data = vectorize_and_save_xvl_data("rgb.xvl", "rgb_vec.json")
    tst_vec_data = vectorize_and_save_xvl_data("rgb_tst.xvl", "rgb_tst_vec.json", palette=lrn_vec_data['palette'])

    lrn_vec_data = xvl.load_from_json("rgb_vec.json")
    tst_vec_data = xvl.load_from_json("rgb_tst_vec.json")

    res_labels = test_learn_predict(lrn_vec_data, tst_vec_data, use_PCA=True)
    print(res_labels)
    xvl.set_labels_to_xvl_color_matrix_file("rgb_tst.xvl", "rgb_tst_res.xvl", res_labels)

    tst_xvl_data = xvl.parse_xvl_color_matrix_file("rgb_tst.xvl")
    mean_labels = mean_rgb_labels(tst_xvl_data)
    print(mean_labels)
    xvl.set_labels_to_xvl_color_matrix_file("rgb_tst.xvl", "rgb_tst_mean.xvl", mean_labels)


if __name__ ==  "__main__":
    # train()
    test()

    # tst_xvl_data = xvl.parse_xvl_color_matrix_file("rgb.xvl")
    # mean_labels = mean_rgb_labels(tst_xvl_data)
    # xvl.set_labels_to_xvl_color_matrix_file("rgb.xvl", "rgb_mean.xvl", mean_labels)
