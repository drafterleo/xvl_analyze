from operator import itemgetter
from math import sqrt
import itertools
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing, LineString
from shapely.ops import cascaded_union
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
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    p = np.zeros((n, 2))
    p[:, 0] = x0 + rx * ct
    p[:, 1] = y0 - ry * st
    return p


# figure: [(x, y), (x, y), ...]
def fig_intersects(fig1, fig2) -> float:
    lr1 = LinearRing(fig1)
    lr2 = LinearRing(fig2)
    return 1 if lr1.intersects(lr2) else 0


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


def fig_contains(fig1, fig2) -> float:
    pgz1 = polygonize_figure(fig1)
    pgz2 = polygonize_figure(fig2)
    if len(pgz1) > 0 and len(pgz2) > 0:
        plist1 = [Polygon(p) for p in pgz1]
        plist2 = [Polygon(p) for p in pgz2]
        u_polygon1 = cascaded_union(plist1)
        u_polygon2 = cascaded_union(plist2)
        if u_polygon1.contains(u_polygon2):
            return True
    return False


# figure: [(x, y), (x, y), ...]
def fig_area(fig) -> float:
    polygons = polygonize_figure(fig)
    if len(polygons) > 0:
        plist = [Polygon(p) for p in polygons]
        u_polygon = cascaded_union(plist)
        area = u_polygon.area
    else:
        area = 0
        print(len(fig))
    return area


# figure: [(x, y), (x, y), ...]
def fig_overlap_area(fig1, fig2) -> float:
    pgzfg1 = polygonize_figure(fig1)
    pgzfg2 = polygonize_figure(fig2)
    if len(pgzfg1) > 0 and len(pgzfg2) > 0:
        plist1 = [Polygon(p) for p in pgzfg1]
        plist2 = [Polygon(p) for p in pgzfg2]
        u_polygon1 = cascaded_union(plist1)
        u_polygon2 = cascaded_union(plist2)
        area = u_polygon1.intersection(u_polygon2).area
    else:
        area = 0.0
    return area


def fig_mosaic_rate(fig) -> float:
    polygons = polygonize_figure(fig)
    return len(polygons)/100


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


def pix_vertex_density(figures, size=3) -> list:
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


def pix_area_density(figures, size=3) -> list:
    dx = dy = 1/size
    cells = []
    for y in range(size):
        for x in range(size):
            cells.append([x*dx, y*dy, x*dx + dx, y*dx + dy])
    density = [0.]*len(cells)
    for fig in figures:
        pgzfg = polygonize_figure(fig)
        if len(pgzfg) > 0:
            plist = [Polygon(p) for p in pgzfg]
            fig_polygon = cascaded_union(plist)
            for i, (x0, y0, x1, y1) in enumerate(cells):
                cell_polygon = shapely.geometry.box(x0, y0, x1, y1)
                density[i] += fig_polygon.intersection(cell_polygon).area
    return density


# fig: [(x, y), (x, y), ...]
def make_point_link_data(fig):
    n = len(fig)
    links = dict([(i, {(i + 1) % n, (i - n - 1) % n}) for i in range(n)])
    cross_points = []
    cross_vertexes = {}
    crossed_lines = []
    cp_idx = n
    # find cross-points
    for i in range(len(fig) - 1):
        for j in range(i + 1, len(fig)):
            edge_vertexes = [i % n, (i + 1) % n, j % n, (j + 1) % n]
            i1, i2, j1, j2 = edge_vertexes
            edge_i = LineString([fig[i1], fig[i2]])
            edge_j = LineString([fig[j1], fig[j2]])
            if edge_i.crosses(edge_j):
                ip = edge_i.intersection(edge_j)
                cross_points.append((ip.x, ip.y))
                links[cp_idx] = set()
                cr_line1 = {i1, i2}
                cr_line2 = {j1, j2}
                cross_vertexes[cp_idx] = (cr_line1, cr_line2)
                if cr_line1 not in crossed_lines:
                    crossed_lines.append(cr_line1)
                if cr_line2 not in crossed_lines:
                    crossed_lines.append(cr_line2)
                cp_idx += 1
    points = fig + cross_points
    # add cross-point links
    for base_line in crossed_lines:
        inner = []  # inline vertexes
        for vx, lines in cross_vertexes.items():
            if base_line in lines:
                inner.append(vx)
        if len(inner) > 0:
            bs1, bs2 = list(base_line)
            if len(inner) >= 2:
                inner.sort(key=lambda x: Point(points[x]).distance(Point(points[bs1])))
            links[bs1].remove(bs2)
            links[bs2].remove(bs1)
            vxs = [bs1] + inner + [bs2]
            for i in range(1, len(vxs) - 1):
                links[vxs[i]].add(vxs[i-1])
                links[vxs[i-1]].add(vxs[i])
                links[vxs[i]].add(vxs[i+1])
                links[vxs[i+1]].add(vxs[i])

    return points, links


# fig: [(x, y), (x, y), ...]
def polygonize_figure(fig):
    if len(fig) < 3 or LinearRing(fig).is_simple:
        return [fig]
    points, links = make_point_link_data(fig)
    cycles = []
    max_depth = len(points)

    # def find_cycles(path, depth):
    #     if depth < max_depth:
    #         curr_vx = path[-1]  # last one
    #         for vx in links[curr_vx]:
    #             if depth > 1 and vx == path[0]:
    #                 path.append(vx)
    #                 cycles.append(path)
    #                 break
    #             elif vx not in path:
    #                 find_cycles(path + [vx], depth + 1)

    def find_cycles(path, depth):
        if depth < max_depth:
            curr_vx = path[-1]  # last one
            if depth > 1 and path[0] in links[curr_vx]:
                cycles.append(path)
            else:
                for vx in links[curr_vx]:
                    if vx not in path:
                        find_cycles(path + [vx], depth + 1)

    for i in range(len(fig), len(points)):
        find_cycles([i], 0)

    cycle_sets = [set(c) for c in cycles]
    cycles_pack = [cycles[i] for i in range(len(cycles))
                   if set(cycles[i]) not in cycle_sets[i+1:]]
    polygons = [[points[idx] for idx in cycle] for cycle in cycles_pack]
    return polygons


def test():
    # points, links, crossed_lines = make_point_link_data([(1., 0.), (5., 0.), (0., 2.), (6., 3.), (1., 5.), (6., 6.)]) # saw
    # points, links, crossed_lines = make_point_link_data([(1., 1.), (8., 4.), (2., 5.), (6., 0.), (8., 2.), (0., 3.)]) # mill
    # points, links, crossed_lines = make_point_link_data([(1., 0.), (7., 3.), (0., 3.), (6., 0.), (4., 5.)])  # star
    # print(links)
    # print(crossed_lines)

    # polygons = polygonize_figure([(0., 0.), (3., 0.), (0., 3.), (0., 5.), (3., 5.), (3., 3.)])  # cup: 2 [10.5]
    # polygons = polygonize_figure([(4., 4.), (0., 0.), (3., 0.), (0., 2.), (4., 2.)])  # 2-saw: 3 [4.6]
    # polygons = polygonize_figure([(1., 0.), (5., 0.), (0., 2.), (6., 3.), (1., 5.), (6., 6.)])  # 3-saw: 4 [10.384]
    # polygons = polygonize_figure([(1., 1.), (8., 4.), (2., 5.), (6., 0.), (8., 2.), (0., 3.)])  # mill: 7 [15.472]
    polygons = polygonize_figure([(1., 0.), (7., 3.), (0., 3.), (6., 0.), (4., 5.)])  # star: 26 [11.891]
    pprint(polygons)
    print(len(polygons))
    plist = [Polygon(p) for p in polygons]
    u_polygon = cascaded_union(plist)
    print(u_polygon.boundary)
    print(u_polygon.area)


if __name__ == "__main__":
    test()