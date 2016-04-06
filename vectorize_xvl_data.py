import parse_xvl as xvl
import color_maps as cmaps
import numpy as np
import colorsys
import itertools
from color_palette import color_names
from pprint import pprint
import figure_utils as fgut


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

def flatten(lst):
    return list(itertools.chain.from_iterable(lst))


def vectorize_xvl_color_data(xvl_data, palette=[]) -> (list, list):
    data = xvl_data
    if palette:
        color_list = list(xvl.xvl_data_color_set(xvl_data))
        color_map = cmaps.make_palette_map(color_list, palette)
        data = [(label, [color_map[color] for color in vector]) # map xvl_data colors to palette
                for label, vector in xvl_data]

    idx_label_map = dict(enumerate(list(set(xvl.labels_of_xvl_data(xvl_data)))))
    label_idx_map = dict([(label, index) for index, label in idx_label_map.items()])
    vec_data = [(label_idx_map[label],
                 flatten([cmaps.hex2rgb(color) for color in vector]))  # RGB's to flat list
                for label, vector in data]
    return vec_data, idx_label_map


def mean_by_indices(rgb_vectors, indices) -> list:
    thin_vectors = [np.array([v[i] for i in indices]) for v in rgb_vectors]
    rgb_mean_vectors = [np.mean(v, axis=0) for v in thin_vectors]
    return rgb_mean_vectors


def make_mean_features(hex_color_vectors) -> list:
    rgb_vectors = [[cmaps.hex2rgb(c) for c in v] for v in hex_color_vectors]
    gnz_indices = ((0, 1, 2, 3, 4, 5, 6, 7, 8),
                   (0, 1, 2), (3, 4, 5), (6, 7, 8),
                   (0, 3, 6), (1, 4, 7), (2, 5, 8),
                   (0, 4, 8), (2, 4, 6),
                   (0, 1, 3, 4), (1, 2, 4, 5), (3, 4, 6, 7), (4, 5, 7, 8),
                   (1, 4, 7, 3, 5))

    features = [[] for _ in range(len(rgb_vectors))]
    for indices in gnz_indices:
        f = mean_by_indices(rgb_vectors, indices)
        for i in range(len(features)):
            #features[i].append(f[i])
            features[i] += list(f[i])
    return features


def rgb_diff(hex_color: str):
    r, g, b = cmaps.hex2rgb(hex_color)
    return abs(r - g), abs(g - b), abs(b - r)


def make_xvl_color_vec_data(xvl_data, palette=[]) -> dict:
    xvl_vec_data, idx_label_map = vectorize_xvl_color_data(xvl_data, palette=palette)
    vectors = xvl.vectors_of_xvl_data(xvl_vec_data)

    # # generalized colors feature
    # _, gnz_palette = cmaps.make_cluster_map(palette, n_clusters=10)
    # xvl_vec_data_gnz, _ = vectorize_xvl_color_data(xvl_data, gnz_palette)
    # gnz_features = xvl.vectors_of_xvl_data(xvl_vec_data_gnz)

    # saturation features
    hex_vectors = xvl.vectors_of_xvl_data(xvl_data)
    hsv_features = [[colorsys.rgb_to_hsv(*cmaps.hex2rgb(c))[0] for c in vec]
                    for vec in hex_vectors]
    diff_features = [flatten([rgb_diff(c) for c in vec])
                     for vec in hex_vectors]
    mean_features = make_mean_features(xvl.vectors_of_xvl_data(xvl_data))
    vectors = [  vectors[i]
               + mean_features[i]
               + hsv_features[i]
               + diff_features[i]
               # + gnz_features[i]
               for i in range(len(vectors))]

    labels = xvl.labels_of_xvl_data(xvl_vec_data)

    return {'labels_map': idx_label_map,
            'vectors': vectors,
            'labels': labels,
            'palette': palette}


# xvl_data: [(label, pixra[figure1, figure2, ...], fig_types), ...]
def make_xvl_figures_vec_data(xvl_data) -> dict:
    labels = [item[0] for item in xvl_data]
    idx_label_map = dict(enumerate(list(set(labels))))

    # item[1][i] - figures
    # item[2][i] - figure types
    pixras = [  [fgut.ellipse_to_polygon(item[1][i]) if item[2][i] == 'CEllipseFigure' else item[1][i]
                 for i in range(len(item[1]))]
              for item in xvl_data]

    fig_types = [item[2] for item in xvl_data]

    coordinate_features = np.array([flatten([flatten(fig) for fig in figs])
                                    for figs in pixras])
    # sizes and centers
    metric_features = np.array([flatten([fgut.fig_metrics(fig) for fig in figs])
                                for figs in pixras])

    overlap_features = np.array([[fgut.fig_intersection(figs[i], figs[i + 1]) for i in range(len(figs) - 1)]
                                 for figs in pixras])

    inner_deltas_features = np.array([flatten([fgut.fig_inner_deltas(fig) for fig in figs])
                               for figs in pixras])

    distance_features = np.array([flatten([fgut.fig_distance(figs[i], figs[i + 1])
                                           for i in range(len(figs) - 1)])
                                  for figs in pixras])

    inner_angles_features = np.array([flatten([fgut.fig_inner_angles(fig) for fig in figs])
                                     for figs in pixras])

    # vectors = overlap_features
    vectors = np.hstack((# inner_deltas_features,
                         metric_features *.05,
                         # coordinate_features,
                         overlap_features,
                         # inner_angles_features,
                         distance_features
    ))

    return {'labels_map': idx_label_map,
            'labels'    : labels,
            'vectors'   : vectors,
            'figures'   : pixras,
            'types'     : fig_types}


def test_color_vectorize():
    xvl_data = xvl.parse_xvl_color_matrix_file("ai_src.xvl")
    xvl_vec_data, idx_labels_map = vectorize_xvl_color_data(xvl_data)
    print(xvl_vec_data)
    print(len(xvl_vec_data))


def test_figures_vectorize():
    xvl_data = xvl.parse_xvl_figures_file("fig_tst.xvl")
    make_xvl_figures_vec_data(xvl_data)


if __name__ == "__main__":
    test_figures_vectorize()




