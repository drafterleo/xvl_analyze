import numpy as np
import re
from sklearn.cluster import KMeans
import parse_xvl as xvl

color_names = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksage':             '#598556',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsage':            '#BCECAC',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sage':                 '#87AE73',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}

def rgb2hex(rgb):
    a = '#%02x%02x%02x' % tuple([int(np.round(val * 255)) for val in rgb[:3]])
    return a

hexColorPattern = re.compile("\A#[a-fA-F0-9]{6}\Z")

def hex2color(s: str):
    if hexColorPattern.match(s) is None:
        raise ValueError('invalid hex color string "%s"' % s)
    return tuple([int(n, 16) / 255.0 for n in (s[1:3], s[3:5], s[5:7])])

def make_cluster_map(colors, num_clusters=100):
    color_vectors = np.array([hex2color(c) for c in colors])

    # initialize a k-means object and use it to extract centroids
    kmeans_model = KMeans(n_clusters=num_clusters)
    idx = kmeans_model.fit_predict(color_vectors)
    print("k-means clustering done")

    color_map = dict([(colors[i], rgb2hex(kmeans_model.cluster_centers_[idx[i]]))
                      for i in range(len(colors))])
    #xvl.show_colors([colors.rgb2hex(i) for i in kmeans_model.cluster_centers_])
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


def nearest_color_vec(color, palette, palette_matrix):
    color_vector = norm_color_vector(color)
    nearest_palette_idx = np.argmax(np.dot(color_vector, palette_matrix.T))
    return palette[nearest_palette_idx] if np.sum(color_vector) > 0 else '#000000'


def nearest_color_rgb(color, palette, palette_rgb):
    color_rgb = np.array(hex2color(color))
    rgb_difference = np.sum(np.abs(palette_rgb - color_rgb), axis=1)
    return palette[np.argmin(rgb_difference)]


def make_palette_map(colors, palette):
    colors = list(xvl.xvl_data_color_set(xvl_data))
    palette_rgb = np.array([hex2color(c) for c in palette])
    # palette_matrix = make_palette_matrix(palette)
    color_map = {}
    for color in colors:
        color_map[color] = nearest_color_rgb(color, palette, palette_rgb)
        # color_map[color] = nearest_color_vec(color, palette, palette_matrix)
    return color_map


if __name__ ==  "__main__":
    xvl_data = xvl.parse_xvl_file("animate_inanimate_colors.xvl")
    color_list = list(xvl.xvl_data_color_set(xvl_data))
    cluster_map, cluster_palette = make_cluster_map(color_list)
    print(cluster_map)

    palette = list(color_names.values())
    palette_map = make_palette_map(color_list, cluster_palette)
    print(palette_map)

    map_diff = [(c, cluster_map[c], palette_map[c])
                for c in color_list if cluster_map[c] != palette_map[c]]
    print(len(map_diff), map_diff)
