import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

# http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree
def parse_xvl_color_matrix_file(file_name) -> list:
    data = []
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()
    if root.tag == 'xravlaste':
        for item in root:
            if item.tag == 'item':
                word_item = item[0]
                pixra_item = item[2]
                if pixra_item.attrib['type'] == 'CColorMatrixPixra':
                    label = word_item.text
                    colors = []
                    matrix_item = pixra_item[0]
                    for color_item in matrix_item:
                       color = color_item.attrib['rgb']
                       colors.append(color)
                    data.append((label, colors))
    return data # [(label, [colors]), ...]


def remap_xvl_color_matrix_file(src_file, dst_file, color_map):
    tree = ET.ElementTree(file=src_file)
    root = tree.getroot()
    if root.tag == 'xravlaste':
        for item in root:
            if item.tag == 'item':
                pixra_item = item[2]
                if pixra_item.attrib['type'] == 'CColorMatrixPixra':
                    matrix_item = pixra_item[0]
                    for color_item in matrix_item:
                       color = color_item.attrib['rgb']
                       color_item.attrib['rgb'] = color_map.get(color, color)
    tree.write(file_or_filename=dst_file)


def show_colors(colors: list):
    fig = plt.figure()
    y = 0
    w = 0.5
    h = 1/len(colors)
    for color in colors:
        pos = (0, y)
        ax = fig.add_subplot(111)
        ax.add_patch(plt.Rectangle(pos, w, h, color=color))
        ax.annotate(color, pos)
        y += h
    plt.show()


def xvl_data_label_set(data):
    labels = set()
    for item in data:
        labels.add(item[0])
    return labels


def labels_of_xvl_data(xvl_data):
    return [item[0] for item in xvl_data]


def vectors_of_xvl_data(xvl_data):
    return [item[1] for item in xvl_data]


def xvl_data_color_set(data):
    colors = set()
    for item in data:
        for color in item[1]:
            colors.add(color)
    return colors


def test_xvl_parser():
    xvl_data = parse_xvl_color_matrix_file("animate_inanimate_colors.xvl")
    print(len(xvl_data), xvl_data)
    print(xvl_data_label_set(xvl_data))
    color_set = xvl_data_color_set(xvl_data)
    print(len(color_set))
    show_colors(xvl_data[0][1])


if __name__ ==  "__main__":
    test_xvl_parser()
