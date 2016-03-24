import parse_xvl as xvl
import color_maps as cmap

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

    xvl_vec_data = [(label_idx_map[label], [color_idx_map[color] for color in vector])
                    for label, vector in xvl_data]

    return xvl_vec_data, idx_palette_map, idx_label_map

def make_xvl_color_matrix_vec_data(xvl_data):
    xvl_vec_data, idx_palette_map, idx_label_map = vectorize_xvl_color_matrix_data(xvl_data, palette_size=150)
    vectors = xvl.vectors_of_xvl_data(xvl_vec_data)

    gnz_palette = ['#FF0000', '#00FF00', '#0000FF',
                   '#FF8800', '#88FF00', '#0088FF',
                   '#FF0088', '#00FF88', '#8800FF']

    # generalized colors
    # xvl_vec_data_gnz, _, _ = vectorize_xvl_color_matrix_data(xvl_data, palette_size=10)
    # vectors_gnz = xvl.vectors_of_xvl_data(xvl_vec_data_gnz)

    vectors_gnz = []
    color_count = len(idx_palette_map)
    for vector in vectors:
        color_vec = [0] * color_count
        for i in vector:
            color_vec[i] = 1
        vectors_gnz.append(color_vec)

    vectors = [vectors[i] + vectors_gnz[i] for i in range(len(vectors))]

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




