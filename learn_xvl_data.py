import numpy as np
import parse_xvl as xvl
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from vectorize_xvl_data import make_xvl_color_matrix_vec_data


def learn_predict(xvl_vec_data):
    vectors = xvl_vec_data['vectors']
    labels = xvl_vec_data['labels']

    print(xvl_vec_data['labels-map'])
    print(vectors[0])

    tst_count = 10

    X = vectors[:-tst_count]
    Y = labels[:-tst_count]

    X_tst = vectors[-tst_count:]
    Y_tst = labels[-tst_count:]

    # classifier = svm.SVC(decision_function_shape='ovo', kernel='linear')
    classifier = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    # classifier = GaussianNB()
    classifier.fit(X, Y)

    print(list(classifier.predict(X_tst)))
    print(Y_tst)


def make_and_save_xvl_vec_data(xvl_file, json_file):
    xvl_data = xvl.parse_xvl_color_matrix_file(xvl_file)
    xvl_vec_data = make_xvl_color_matrix_vec_data(xvl_data)
    xvl.save_to_json(xvl_vec_data, json_file)
    return xvl_vec_data

if __name__ ==  "__main__":
    #xvl_vec_data = make_and_save_xvl_vec_data("ai_src.xvl", "ai_src_vec.json")

    xvl_vec_data = xvl.load_from_json("ai_src_vec.json")
    learn_predict(xvl_vec_data)
