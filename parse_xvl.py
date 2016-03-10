import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

# http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree
def parse_xvl_file(file_name) -> list:
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

def xvl_data_labels(data):
    labels = set()
    for item in data:
        labels.add(item[0])
    return labels


def xvl_data_color_set(data):
    colors = set()
    for item in data:
        for color in item[1]:
            colors.add(color)
    return colors


if __name__ ==  "__main__":
    #show_colors(colors=['#D2B48C', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3'])
    xvl_data = parse_xvl_file("animate_inanimate_colors.xvl")
    print(len(xvl_data), xvl_data)
    print(xvl_data_labels(xvl_data))
    color_set = xvl_data_color_set(xvl_data)
    print(len(color_set))
    show_colors(xvl_data[0][1])
