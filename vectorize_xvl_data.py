import parse_xvl as xvl
import color_maps as cmaps
import numpy as np
import itertools
import colorsys
from operator import itemgetter
from math import sqrt
from color_palette import color_names
from pprint import pprint


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


# figure: [(x, y), (x, y), ...]
def fig_center(figure: list) -> (float, float):
    min_x = min(figure, key=itemgetter(0))[0]
    min_y = min(figure, key=itemgetter(1))[1]
    max_x = max(figure, key=itemgetter(0))[0]
    max_y = max(figure, key=itemgetter(1))[1]
    center_x = min_x + (max_x - min_x)/2
    center_y = min_y + (max_y - min_y)/2
    return center_x, center_y


def fig_distance(fig1, fig2) -> (float, float):
    center1 = fig_center(fig1)
    center2 = fig_center(fig2)
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    return dx, dy


# figure: [(x, y), (x, y), ...]
def fig_deltas(fig):
    deltas = [(fig[i][0] - fig[i + 1][0], fig[i][1] - fig[i + 1][1])
              for i in range(len(fig) - 1)]
    return deltas


# figure: [(x, y), (x, y), ...]
def fig_angles(fig) -> list:
    angles = []
    ext_fig = [fig[-1]] + fig + [fig[0]]
    for i in range(1, len(ext_fig) - 1):
        ax, ay = ext_fig[i]       # curr vertex
        px, py = ext_fig[i - 1]   # prev vertex
        nx, ny = ext_fig[i + 1]   # next vertex
        nx -= ax; px -= ax
        py -= ay; ny -= ay
        cos_a = (px*nx + py*ny)/sqrt((px**2 + py**2) * (nx**2 + ny**2))
        angles.append(cos_a)
    return angles


# xvl_data: [(label, pixra[figure1, figure2, ...], fig_types), ...]
def make_xvl_figures_vec_data(xvl_data) -> dict:
    labels = [item[0] for item in xvl_data]
    idx_label_map = dict(enumerate(list(set(labels))))

    pixras = [item[1] for item in xvl_data]
    fig_types = [item[2] for item in xvl_data]

    # coordinates
    coordinate_features = np.array([flatten([flatten(fig) for fig in figs])
                                    for figs in pixras])
    # dx, dy between anchors
    delta_features = np.array([flatten([flatten(fig_deltas(fig)) for fig in figs])
                               for figs in pixras])
    # distances between figures
    distance_features = np.array([flatten([fig_distance(figs[i], figs[i + 1])
                                           for i in range(len(figs) - 1)])
                                  for figs in pixras])
    # angles in figure
    angle_features = np.array([flatten([fig_angles(fig) for fig in figs])
                               for figs in pixras])

    vectors = np.hstack((#delta_features,
                         coordinate_features,
                         distance_features * 5,
                         angle_features))
    # vectors = [distance_features[i]
    #            + delta_features[i]
    #            + angle_features[i]
    #            for i in range(len(pixras))]

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




