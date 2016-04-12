from operator import itemgetter
from math import sqrt
import itertools
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing, LineString
from shapely.ops import polygonize


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


# figure: [(x, y), (x, y), ...]
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


# figure: [(x, y), (x, y), ...]
def fig_intersects(fig1, fig2) -> float:
    ls1 = LineString(fig1)
    ls2 = LineString(fig2)
    return 1 if ls1.intersects(ls2) else 0


def fig_inner_cross_count(fig) -> int:
    count = 0
    ext_fig = fig + [fig[0]]
    for i in range(len(ext_fig) - 1):
        for j in range(i + 1, len(ext_fig) - 1):
            edge_i = LineString([ext_fig[i], ext_fig[i+1]])
            edge_j = LineString([ext_fig[j], ext_fig[j+1]])
            if edge_i.crosses(edge_j):
                count += 1
    return count


# figure: [(x, y), (x, y), ...]
def fig_contains(fig1, fig2) -> float:
    ls1 = LineString(fig1)
    ls2 = LineString(fig2)
    if ls1.is_ring and ls2.is_ring:
        p1 = Polygon(LinearRing(fig1))
        p2 = Polygon(LinearRing(fig2))
        result = p1.contains(p2)
    else:
        result = False
    return 1 if result else 0


# figure: [(x, y), (x, y), ...]
def fig_overlap_area(fig1, fig2) -> float:
    ls1 = LineString(fig1)
    ls2 = LineString(fig2)
    if ls1.is_ring and ls2.is_ring:
        p1 = Polygon(LinearRing(fig1))
        p2 = Polygon(LinearRing(fig2))
        area = p1.intersection(p2).area
    else:
        area = 0.0
    return area


# figure: [(x, y), (x, y), ...]
def fig_area(fig) -> float:
    if LineString(fig).is_ring:
        p = Polygon(LinearRing(fig))
        area = p.area
    else:
        min_x, min_y, max_x, max_y = fig_rect(fig)
        size_x = max_x - min_x
        size_y = max_y - min_y
        area = size_x * size_y
    return area


# figure: [(x, y), (x, y), ...]
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


# figure: [(x, y), (x, y), ...]
def fig_distance(fig1, fig2) -> float:
    center1 = fig_center(fig1)
    center2 = fig_center(fig2)
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    return sqrt(dx**2 + dy**2)


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


# figure: [(x, y), (x, y), ...]
def fig_metrics(fig) -> (float, float, float, float):
    min_x, min_y, max_x, max_y = fig_rect(fig)
    size_x = max_x - min_x
    size_y = max_y - min_y
    center_x = min_x + size_x/2
    center_y = min_y + size_y/2
    return size_x, size_y, center_x, center_y


def pix_density(figures, size=3) -> list:
    dx = dy = 1/size
    cells = []
    for y in range(size):
        for x in range(size):
            cells.append([x*dx, y*dy, x*dx + dx, y*dx + dy])
    density = [0]*len(cells)
    for vx, vy in flatten(figures):
        for i, (x0, y0, x1, y1) in enumerate(cells):
            if (x0 <= vx < x1) and (y0 <= vy < y1):
                density[i] += 1
    return density


def polygonize_figure(fig):
    ext_fig = fig + [fig[0]]
    lines = []
    for i in range(len(ext_fig) - 1):
        lines.append((ext_fig[i], ext_fig[i+1]))
        for j in range(i + 1, len(ext_fig) - 1):
            edge_i = LineString([ext_fig[i], ext_fig[i+1]])
            edge_j = LineString([ext_fig[j], ext_fig[j+1]])
            if edge_i.crosses(edge_j):
                ip = edge_i.intersection(edge_j)
                px, py = ip[0].coors
                lines.append()




def test():
    from pprint import pprint
    lines = (((0, 0), (4, 4)),
             ((4, 4), (0, 4)),
             ((0, 4), (4, 0)),
             ((4, 0), (0, 0)),
             ((0, 0), (2, 2)),
             ((4, 4), (2, 2)),
             ((0, 4), (2, 2)),
             ((4, 0), (2, 2))
             )
    pp = polygonize(lines)
    pprint(list(pp))
    ls = LineString([(0, 0), (1, 1), (0, 1)])


if __name__ == "__main__":
    test()