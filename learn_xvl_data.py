import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import parse_xvl as xvl
from vectorize_xvl_data import make_xvl_color_matrix_vec_data


def learn_predict(xvl_vec_data):
    vectors = xvl_vec_data['vectors']
    labels = xvl_vec_data['labels']

    print(xvl_vec_data['labels-map'])
    # print(vectors[0])

    tst_count = 10

    X = vectors[:-tst_count]
    Y = labels[:-tst_count]

    X_tst = vectors[-tst_count:]
    Y_tst = labels[-tst_count:]

    # classifier = svm.SVC(decision_function_shape='ovo', kernel='linear')
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    classifier = LinearDiscriminantAnalysis(solver='svd')
    # classifier = GaussianNB()
    classifier.fit(X, Y)

    print(list(classifier.predict(X_tst)))
    print(Y_tst)
    print(classifier.fit(X, Y).score(X_tst, Y_tst))

    # show_pca_transform(X, Y)


def show_pca_transform(X, Y):
    pca = decomposition.PCA(n_components=3)
    pca.fit(X, Y)
    X_pca = pca.transform(X)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=Y)
    plt.show()



def make_and_save_xvl_vec_data(xvl_file, json_file):
    xvl_data = xvl.parse_xvl_color_matrix_file(xvl_file)
    xvl_vec_data = make_xvl_color_matrix_vec_data(xvl_data)
    xvl.save_to_json(xvl_vec_data, json_file)
    return xvl_vec_data

if __name__ ==  "__main__":
    # xvl_vec_data = make_and_save_xvl_vec_data("ai_src.xvl", "ai_src_vec.json")

    xvl_vec_data = xvl.load_from_json("ai_src_vec.json")
    learn_predict(xvl_vec_data)
