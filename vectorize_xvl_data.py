import parse_xvl as xvl
import color_maps as cmap
import numpy as np
import itertools
from color_palette import color_names


def vectorize_xvl_color_matrix_data(xvl_data, palette_size=150, base_palette=[]):
    # form palette and color map
    color_list = list(xvl.xvl_data_color_set(xvl_data))

    if len(base_palette) == 0:
        _, palette = cmap.make_cluster_map(color_list, n_clusters=palette_size)
    else:
        palette = base_palette

    color_map = cmap.make_palette_map(color_list, palette)

    idx_palette_map = dict(enumerate(palette))
    idx_label_map = dict(enumerate(list(set(xvl.labels_of_xvl_data(xvl_data)))))

    palette_idx_map = dict([(color, index) for index, color in idx_palette_map.items()])
    label_idx_map = dict([(label, index) for index, label in idx_label_map.items()])

    color_idx_map = dict([(color, palette_idx_map[color_map[color]])
                          for color in color_map.keys()])

    # xvl_vec_data = [(label_idx_map[label], [color_idx_map[color] for color in vector])
    #                 for label, vector in xvl_data]

    xvl_vec_data = [(label_idx_map[label],
                     list(itertools.chain.from_iterable([cmap.hex2color(color) for color in vector]))) # RGB's to flat list
                    for label, vector in xvl_data]

    return xvl_vec_data, idx_palette_map, idx_label_map


def mean_by_indices(rgb_vectors, indices):
    thin_vectors = [np.array([v[i] for i in indices]) for v in rgb_vectors]
    rgb_mean_vectors = [np.mean(v, axis=0) for v in thin_vectors]
    #feature = [int(rgb[0] * 100 + rgb[1] * 10 + rgb[2]) for rgb in rgb_mean_vectors]
    return rgb_mean_vectors


def mean_features(color_vectors):
    rgb_vectors = [[cmap.hex2color(c) for c in v] for v in color_vectors]
    gnz_indices = ((0, 1, 2, 3, 4, 5, 6, 7, 8),
                   (0, 1, 2), (3, 4, 5), (6, 7, 8),
                   (0, 3, 6), (1, 4, 7), (2, 5, 8),
                   (0, 4, 8), (2, 4, 6),
                   (0, 1, 3, 4), (1, 2, 4, 5), (3, 4, 6, 7), (4, 5, 7, 8))

    features = [[] for _ in range(len(rgb_vectors))]
    for indices in gnz_indices:
        f = mean_by_indices(rgb_vectors, indices)
        for i in range(len(features)):
            #features[i].append(f[i])
            features[i] += list(f[i])
    return features


def make_xvl_color_matrix_vec_data(xvl_data):
    xvl_vec_data, idx_palette_map, idx_label_map = vectorize_xvl_color_matrix_data(xvl_data, palette_size=150)
    vectors = xvl.vectors_of_xvl_data(xvl_vec_data)


    gnz_palette = ['#FF0000', '#00FF00', '#0000FF',
                   '#FF8800', '#88FF00', '#0088FF',
                   '#FF0088', '#00FF88', '#8800FF']

    # generalized colors
    xvl_vec_data_gnz, _, _ = vectorize_xvl_color_matrix_data(xvl_data, palette_size=10)
    features_gnz = xvl.vectors_of_xvl_data(xvl_vec_data_gnz)

    # features = []
    # color_count = len(idx_palette_map)
    # for vector in vectors:
    #     color_vec = [0] * color_count
    #     for i in vector:
    #         color_vec[i] = 1
    #     features.append(color_vec)

    features = mean_features(xvl.vectors_of_xvl_data(xvl_data))

    vectors = [vectors[i] + features[i] + features_gnz[i] for i in range(len(vectors))]

    labels = xvl.labels_of_xvl_data(xvl_vec_data)

    return {'labels_map': idx_label_map,
            'palette_map': idx_palette_map,
            'vectors': vectors,
            'labels': labels}


def test():
    xvl_data = xvl.parse_xvl_color_matrix_file("ai_src.xvl")
    xvl_vec_data, idx_palette_map, idx_labels_map = vectorize_xvl_color_matrix_data(xvl_data, palette_size=150)
    print(idx_palette_map)
    print(xvl_vec_data)
    print(len(xvl_vec_data))

    xvl_vec_item = xvl_vec_data[0]
    remap = [idx_palette_map[idx] for idx in xvl_vec_item[1]]
    print(xvl_data[0][1])
    print(remap)


if __name__ ==  "__main__":
    test()




