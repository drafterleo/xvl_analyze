import numpy as np
import parse_xvl as xvl
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from vectorize_xvl_data import vectorize_xvl_color_matrix_data

xvl_data = xvl.parse_xvl_color_matrix_file("ai_src.xvl")
xvl_vec_data, idx_palette_map, idx_label_map = vectorize_xvl_color_matrix_data(xvl_data, palette_size=150)
xvl_vec_data_gnz, _, _ = vectorize_xvl_color_matrix_data(xvl_data, palette_size=10)

print(idx_label_map)

vectors = xvl.vectors_of_xvl_data(xvl_vec_data)
vectors_gnz = xvl.vectors_of_xvl_data(xvl_vec_data_gnz)
vectors = [vectors[i] + vectors_gnz[i] for i in range(len(vectors))]

labels = xvl.labels_of_xvl_data(xvl_vec_data)

tst_count = 10

X = vectors[:-tst_count]
Y = labels[:-tst_count]

# classifier = svm.SVC(decision_function_shape='ovo', kernel='rbf')
classifier = neighbors.KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
# classifier = BernoulliNB()
classifier.fit(X, Y)

print(classifier.predict(vectors[-tst_count:]))
print(labels[-tst_count:])
