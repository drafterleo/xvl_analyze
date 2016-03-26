import parse_xvl as xvl
import color_maps as cmap
import numpy as np
import itertools
from color_palette import color_names


# def vectorize_xvl_color_matrix_data(xvl_data, palette_size=150, base_palette=[]):
#     # form palette and color map
#     color_list = list(xvl.xvl_data_color_set(xvl_data))
#
#     if len(base_palette) == 0:
#         _, palette = cmap.make_cluster_map(color_list, n_clusters=palette_size)
#     else:
#         palette = base_palette
#
#     color_map = cmap.make_palette_map(color_list, palette)
#
#     idx_palette_map = dict(enumerate(palette))
#     idx_label_map = dict(enumerate(list(set(xvl.labels_of_xvl_data(xvl_data)))))
#
#     palette_idx_map = dict([(color, index) for index, color in idx_palette_map.items()])
#     label_idx_map = dict([(label, index) for index, label in idx_label_map.items()])
#
#     # color_idx_map = dict([(color, palette_idx_map[color_map[color]])
#     #                       for color in color_map.keys()])
#
#     # xvl_vec_data = [(label_idx_map[label], [color_idx_map[color] for color in vector])
#     #                 for label, vector in xvl_data]
#
#     xvl_vec_data = [(label_idx_map[label],
#                      list(itertools.chain.from_iterable([cmap.hex2color(color) for color in vector]))) # RGB's to flat list
#                     for label, vector in xvl_data]
#
#     return xvl_vec_data, idx_palette_map, idx_label_map


def vectorize_xvl_color_matrix_data(xvl_data, palette=[]):
    data = xvl_data
    if palette:
        color_list = list(xvl.xvl_data_color_set(xvl_data))
        color_map = cmap.make_palette_map(color_list, palette)
        data = [(label, [color_map[color] for color in vector]) # map xvl_data colors to palette
                for label, vector in xvl_data]

    idx_label_map = dict(enumerate(list(set(xvl.labels_of_xvl_data(xvl_data)))))
    label_idx_map = dict([(label, index) for index, label in idx_label_map.items()])
    vec_data = [(label_idx_map[label],
                 list(itertools.chain.from_iterable([cmap.hex2rgb(color) for color in vector])))  # RGB's to flat list
                for label, vector in data]
    return vec_data, idx_label_map


def mean_by_indices(rgb_vectors, indices):
    thin_vectors = [np.array([v[i] for i in indices]) for v in rgb_vectors]
    rgb_mean_vectors = [np.mean(v, axis=0) for v in thin_vectors]
    return rgb_mean_vectors


def make_mean_features(hex_color_vectors):
    rgb_vectors = [[cmap.hex2rgb(c) for c in v] for v in hex_color_vectors]
    gnz_indices = ((0, 1, 2, 3, 4, 5, 6, 7, 8),
                   (0, 1, 2), (3, 4, 5), (6, 7, 8),
                   (0, 3, 6), (1, 4, 7), (2, 5, 8),
                   (0, 4, 8), (2, 4, 6),
                   (0, 1, 3, 4), (1, 2, 4, 5), (3, 4, 6, 7), (4, 5, 7, 8),
                   (0, 3, 6, 2, 5, 8), (0, 1, 2, 6, 7, 8),
                   (1, 4, 7, 3, 5))

    features = [[] for _ in range(len(rgb_vectors))]
    for indices in gnz_indices:
        f = mean_by_indices(rgb_vectors, indices)
        for i in range(len(features)):
            #features[i].append(f[i])
            features[i] += list(f[i])
    return features



def make_xvl_color_matrix_vec_data(xvl_data, palette=[]):
    xvl_vec_data, idx_label_map = vectorize_xvl_color_matrix_data(xvl_data, palette=palette)
    vectors = xvl.vectors_of_xvl_data(xvl_vec_data)

    # generalized colors feature
    color_list = list(xvl.xvl_data_color_set(xvl_data))
    _, gnz_palette = cmap.make_cluster_map(palette, n_clusters=10)
    xvl_vec_data_gnz, _ = vectorize_xvl_color_matrix_data(xvl_data, gnz_palette)
    gnz_features = xvl.vectors_of_xvl_data(xvl_vec_data_gnz)

    # features = []
    # color_count = len(idx_palette_map)
    # for vector in vectors:
    #     color_vec = [0] * color_count
    #     for i in vector:
    #         color_vec[i] = 1
    #     features.append(color_vec)

    mean_features = make_mean_features(xvl.vectors_of_xvl_data(xvl_data))

    vectors = [vectors[i] + mean_features[i] #+ gnz_features[i]
               for i in range(len(vectors))]

    labels = xvl.labels_of_xvl_data(xvl_vec_data)

    return {'labels_map': idx_label_map,
            'vectors': vectors,
            'labels': labels,
            'palette': palette}


def test():
    xvl_data = xvl.parse_xvl_color_matrix_file("ai_src.xvl")
    xvl_vec_data, idx_palette_map, idx_labels_map = vectorize_xvl_color_matrix_data(xvl_data)
    print(idx_palette_map)
    print(xvl_vec_data)
    print(len(xvl_vec_data))

    xvl_vec_item = xvl_vec_data[0]
    remap = [idx_palette_map[idx] for idx in xvl_vec_item[1]]
    print(xvl_data[0][1])
    print(remap)


if __name__ ==  "__main__":
    test()




