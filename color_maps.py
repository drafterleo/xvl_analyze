import numpy as np
import re
from sklearn.cluster import KMeans
from color_palette import color_names
import parse_xvl as xvl


# color compare algorithm types
caRgbDistance = 1
caCosSimilarity = 2

def rgb2hex(rgb):
    a = '#%02x%02x%02x' % tuple([int(np.round(val * 255)) for val in rgb[:3]])
    return a

hexColorPattern = re.compile("\A#[a-fA-F0-9]{6}\Z")

def hex2color(s: str):
    if hexColorPattern.match(s) is None:
        raise ValueError('invalid hex color string "%s"' % s)
    return tuple([int(n, 16) / 255.0 for n in (s[1:3], s[3:5], s[5:7])])


def make_cluster_map(colors, n_clusters=100):
    color_vectors = np.array([hex2color(c) for c in colors])

    # initialize a k-means object and use it to extract centroids
    kmeans_model = KMeans(n_clusters=n_clusters)
    idx = kmeans_model.fit_predict(color_vectors)
    print("k-means clustering done")

    color_map = dict([(colors[i], rgb2hex(kmeans_model.cluster_centers_[idx[i]]))
                      for i in range(len(colors))])
    palette = [rgb2hex(c) for c in kmeans_model.cluster_centers_]
    return color_map, palette


def norm_color_vector(hex_color):
    vector =  np.array(hex2color(hex_color))
    return vector / np.linalg.norm(vector) if np.sum(vector) > 0 else vector


def make_palette_matrix(palette):
    palette = [c if c != '#000000' else '#010101' for c in palette]
    matrix = []
    for color in palette:
        vector = np.array(hex2color(color))    # vectorize palette color hex->rgb
        if np.sum(vector) > 0:
            norm = np.linalg.norm(vector)
            matrix.append(vector/norm)         # normalize palette vector
    return np.array(matrix)


def nearest_color_cossim(color, palette, palette_matrix):
    color_vector = norm_color_vector(color)
    nearest_palette_idx = np.argmax(np.dot(palette_matrix, color_vector.T))
    return palette[nearest_palette_idx] if np.sum(color_vector) > 0 else '#000000'


def nearest_color_rgb(color, palette, palette_rgb):
    color_rgb = np.array(hex2color(color))
    rgb_distance = np.sqrt(np.sum((palette_rgb - color_rgb) ** 2, axis=1)) # np.sum(np.abs(palette_rgb - color_rgb), axis=1)
    return palette[np.argmin(rgb_distance)]


def make_palette_map(colors, palette, compare_alg=caRgbDistance):
    colors = list(xvl.xvl_data_color_set(xvl_data))
    color_map = {}
    if compare_alg == caRgbDistance:
        palette_rgb = np.array([hex2color(c) for c in palette])
        for color in colors:
            color_map[color] = nearest_color_rgb(color, palette, palette_rgb)
    else: # caCosSimilarity
        palette_matrix = make_palette_matrix(palette)
        for color in colors:
            color_map[color] = nearest_color_cossim(color, palette, palette_matrix)
    return color_map


if __name__ ==  "__main__":
    xvl_data = xvl.parse_xvl_file("animate_inanimate_colors.xvl")
    color_list = list(xvl.xvl_data_color_set(xvl_data))

    cluster_map, cluster_palette = make_cluster_map(color_list, n_clusters=len(color_list)//9)
    print(cluster_map)

    palette = list(color_names.values())
    palette_map = make_palette_map(color_list, cluster_palette)
    print(palette_map)

    map_diff = [(c, cluster_map[c], palette_map[c])
                for c in color_list if cluster_map[c] != palette_map[c]]
    print(len(map_diff), map_diff)

    xvl.remap_xvl_file("animate_inanimate_colors.xvl", "ai_tst.xvl", palette_map)

    # http://www.color-hex.com/
