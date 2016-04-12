from operator import itemgetter
from math import sqrt
import itertools
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing, LineString
from shapely.ops import polygonize
from pprint import pprint


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


# fig: [(x, y), (x, y), ...]
def make_point_link_data(fig):
    n = len(fig)
    links = dict([(i, [(i + 1) % n, (i - n - 1)%n]) for i in range(n)])
    cross_points = []
    cross_lines = []
    cp_idx = n
    ext_fig = fig + [fig[0]]
    for i in range(len(ext_fig) - 1):
        for j in range(i + 1, len(ext_fig) - 1):
            edge_i = LineString([ext_fig[i], ext_fig[i + 1]])
            edge_j = LineString([ext_fig[j], ext_fig[j + 1]])
            if edge_i.crosses(edge_j):
                ip = edge_i.intersection(edge_j)
                cross_points.append((ip.x, ip.y))
                cross_lines.append({i, i+1})
                cross_lines.append({j, j+1})
                links[cp_idx] = [i, i+1, j, j+1]
                links[i].append(cp_idx)
                links[i+1].append(cp_idx)
                links[j].append(cp_idx)
                links[j+1].append(cp_idx)
                cp_idx += 1
    for i in range(n, cp_idx - 1):
        for j in range(i + 1, cp_idx):
            if len(set(links[i]) & set(links[j])) >= 2:
                links[i].append(j)
                links[j].append(i)
    points = fig + cross_points
    return points, links, cross_lines


# fig: [(x, y), (x, y), ...]
def polygonize_figure(fig):
    points, links, cross_lines = make_point_link_data(fig)
    print(points)
    print(links)
    cycles = []
    max_depth = len(points)

    def find_cycles(base_vx, path, depth):
        if depth < max_depth:
            curr_vx = path[-1]
            for vx in links[curr_vx]:
                if depth > 1 and vx == base_vx and {curr_vx, vx} not in cross_lines:
                    path.append(vx)
                    point_list = [points[idx] for idx in path]
                    ls = LineString(point_list)
                    if ls.is_simple:
                        cycles.append(path)
                        break
                elif vx not in path and {curr_vx, vx} not in cross_lines:
                    find_cycles(base_vx, path + [vx], depth + 1)

    for i in range(len(points)):
        find_cycles(i, [i], 0)

    cycle_sets = [set(c) for c in cycles]
    cycles_pack = [cycles[i] for i in range(len(cycles))
                   if set(cycles[i]) not in cycle_sets[i+1:]]
    polygons = [[points[idx] for idx in cycle] for cycle in cycles_pack]
    return polygons


def test():
    polygons = polygonize_figure([(4., 4.), (0., 0.), (3., 0.), (0., 2.), (4., 2.)])
    pprint(polygons)

#     lines = (((0, 0), (4, 4)),
#              ((4, 4), (0, 4)),
#              ((0, 4), (4, 0)),
#              ((4, 0), (0, 0)),
#              ((0, 0), (2, 2)),
#              ((4, 4), (2, 2)),
#              ((0, 4), (2, 2)),
#              ((4, 0), (2, 2)))
#     polygons = polygonize(lines)
#     polygons = polygonize_figure([(0.0, 0.0), (4.0, 4.0), (0.0, 4.0), (4.0, 0.0)])
    # for polygon in polygons:
    #     print(list(polygon.boundary.coords))


if __name__ == "__main__":
    test()