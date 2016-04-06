from operator import itemgetter
from math import sqrt
import itertools
import numpy as np
from shapely.geometry.polygon import LinearRing


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

# def ellipse_to_polygon(fig, angle=0, n=4):
#     min_x, min_y, max_x, max_y = fig_rect(fig)
#     rx = max_x - min_x
#     ry = max_y - min_y
#     x0 = min_x + rx/2
#     y0 = min_y + ry/2
#     t = np.linspace(0, 2*np.pi, n, endpoint=False)
#     st = np.sin(t)
#     ct = np.cos(t)
#     angle = np.deg2rad(angle)
#     sa = np.sin(angle)
#     ca = np.cos(angle)
#     p = np.empty((n, 2))
#     p[:, 0] = x0 + rx * ca * ct + ry * sa * st
#     p[:, 1] = y0 + rx * sa * ct - ry * ca * st
#     print(fig, (min_x, min_y, rx, ry))
#     print(p)
#     return p


def ellipse_to_polygon(fig, n=10):
    min_x, min_y, max_x, max_y = fig_rect(fig)
    rx = (max_x - min_x)/2
    ry = (max_y - min_y)/2
    x0 = min_x + rx
    y0 = min_y + ry
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    p = np.zeros((n, 2))
    p[:, 0] = x0 + rx * ct
    p[:, 1] = y0 - ry * st
    return p


def fig_intersection(fig1, fig2) -> float:
    e1 = LinearRing(fig1)
    e2 = LinearRing(fig2)
    mp = e1.intersection(e2)
    print("intersection", mp, mp.area)
    # x = [p.x for p in mp]
    # y = [p.y for p in mp]
    return 1 if len(mp) > 0 else 0 #mp.area


def fig_rect(fig) -> (float, float, float, float):
    min_x = min(fig, key=itemgetter(0))[0]
    min_y = min(fig, key=itemgetter(1))[1]
    max_x = max(fig, key=itemgetter(0))[0]
    max_y = max(fig, key=itemgetter(1))[1]
    return min_x, min_y, max_x, max_y


# figure: [(x, y), (x, y), ...]
def fig_center(fig: list) -> (float, float):
    min_x, min_y, max_x, max_y = fig_rect(fig)
    center_x = min_x + (max_x - min_x)/2
    center_y = min_y + (max_y - min_y)/2
    return center_x, center_y


def fig_distance(fig1, fig2) -> (float, float):
    center1 = fig_center(fig1)
    center2 = fig_center(fig2)
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    return sqrt(dx**2 + dy**2), 0.0


# figure: [(x, y), (x, y), ...]
def fig_inner_deltas(fig) -> list:
    deltas = flatten([(fig[i][0] - fig[i + 1][0], fig[i][1] - fig[i + 1][1])
                      for i in range(len(fig) - 1)])
    # deltas = [sqrt((fig[i][0] - fig[i + 1][0])**2 + (fig[i][1] - fig[i + 1][1])**2)
    #           for i in range(len(fig) - 1)]
    return deltas


# figure: [(x, y), (x, y), ...]
def fig_inner_angles(fig) -> list:
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


def fig_metrics(fig) -> (float, float, float, float):
    min_x, min_y, max_x, max_y = fig_rect(fig)
    size_x = max_x - min_x
    size_y = max_y - min_y
    center_x = min_x + size_x/2
    center_y = min_y + size_y/2
    return size_x, size_y, center_x, center_y


def fig_overlap(fig1, fig2) -> (float, float):
    min_x1, min_y1, max_x1, max_y1 = fig_rect(fig1)
    min_x2, min_y2, max_x2, max_y2 = fig_rect(fig2)
    w1 = abs(max_x1 - min_x1)
    h1 = abs(max_y1 - min_y1)
    x_overlap = w1 - abs(min_x1 - min_x2)
    y_overlap = h1 - abs(min_y1 - min_y2)
    return x_overlap, y_overlap
