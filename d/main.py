import numpy as np
from PIL import Image
from scipy.ndimage.measurements import label
from scipy.spatial import ConvexHull, distance
from itertools import product as iter_product
from time import time

np.set_printoptions(threshold=np.inf)
DIST_THRESHOLD = np.e

def find_single_barycenter(pts):
    '''
        (!) "pts" has to be in shape (2, N)
    '''
    r_mean = pts.T[0].mean()
    c_mean = pts.T[1].mean()
    return np.asarray([r_mean, c_mean], dtype=np.float32)

def find_barycenters(pixels):
    barycenters = np.asarray([find_single_barycenter(pts) for pts in pixels], dtype=np.float32)
    return barycenters

def walk_from_barycenter(b_row, b_col, radius, img):
    '''
        How many times color change happen from black to white and vice versa
    '''
    row = b_row; col = b_col
    last = 0; changes = 0
    while col - b_col <= radius:
        if img[row][col] != last:
            changes = changes + 1
        last = img[row][col]
        col = col + 1
    return changes

def prepare_image(path):
    img = np.asarray(Image.open(path), dtype=np.int32)
    # binarizing
    threshold = 127
    img[img <= threshold] = 0
    img[img > threshold] = 255
    labeled, num_shapes = label(img, structure=np.ones((3, 3), dtype=np.int32))
    pixels = [None] * num_shapes

    for idx in range(num_shapes):
        rows, cols = np.where(labeled == idx + 1)
        pixels[idx] = np.vstack((rows, cols)).T

    barycenters = find_barycenters(pixels)
    return img, pixels, barycenters

def process_src_image():
    '''
        Load image, calculate barycenter and classify shapes
    '''
    path = input()
    img, pixels, barycenters = prepare_image(path)
    size = np.asarray([pixel_part.shape[0] for pixel_part in pixels], dtype=np.int32)
    shape = [''] * len(pixels)

    for i, pts in enumerate(pixels):
        ch = ConvexHull(pts)
        ch_idx = np.asarray(ch.vertices, dtype=np.int)
        ch_pts = np.asarray([pts[i] for i in ch_idx], dtype=np.float32)
        b_row = np.int(np.round(barycenters[i][0], 0))
        b_col = np.int(np.round(barycenters[i][1], 0))
        pts_cnt = pts.shape[0]

        if ch_pts.shape[0] >= 30:
            # very probable circle shape
            radius = np.int(np.sqrt(ch.volume / np.pi)) - 1
            if img[b_row][b_col] != 0:
                # only circle has white barycenter
                shape[i] = 'circle'
            else:
                # now color change comes into play; donut has only one
                changes = walk_from_barycenter(b_row, b_col, radius, img)
                shape[i] = 'donut' if changes == 1 else 'flower'
            
        elif img[b_row][b_col] != 0:
            # white barycenter and small CH; either star or cross
            # use volume and pixel_count ratio; cross has low ratio
            ratio = float(pts_cnt / ch.volume)
            shape[i] = 'cross' if ratio < 0.25 else 'star'
        else:
            # only spiral left
            shape[i] = 'spiral'
    
    return barycenters, size, shape

def process_dst_image():
    '''
        Load image, calculate barycenters
    '''
    path = input()
    _, pixels, barycenters = prepare_image(path)
    size = np.asarray([pixel_part.shape[0] for pixel_part in pixels], dtype=np.int32)
    return barycenters, size

def find_affine(pts):
    '''
        Only three points are enough to determine unique affine mapping
    '''
    # (1) M * [0, 0, 1].T
    # (2) M * [1, 0, 1].T
    # (3) M * [0, 1, 1].T
    tx = pts[0][0]; ty = pts[0][1]
    a = pts[1][0] - tx; c = pts[1][1] - ty
    b = pts[2][0] - tx; d = pts[2][1] - ty
    return np.asarray([[a, b, tx], [c, d, ty], [0, 0, 1]], dtype=np.float32)

def is_affine_valid(pts_src_mapped, pts_dst_ext):
    '''
        Affine mapping is valid iff shapes and barycenters match correctly
    '''
    pairwise_dist = distance.cdist(pts_src_mapped, pts_dst_ext)
    # checking if points are far away
    if np.max(np.min(pairwise_dist, axis=0)) > DIST_THRESHOLD:
        return False, None
    mapping = np.argmin(pairwise_dist, axis=0)
    # shapes have to match as well
    size = np.asarray([size_src[idx] for idx in mapping], dtype=np.int32)
    return (size == size_mapping).all(), mapping

def affine_tryout(invM, pts_src, pts_dst, triangle_dst):
    '''
        Brute force for finding correct affine mapping
        Biggest triangle maps to biggest triangle (what a Sherlock)
    '''
    n_pts = pts_src.shape[0]
    pts_src_ext = np.hstack((pts_src, np.ones(shape=(n_pts, 1), dtype=np.float32)))
    pts_dst_ext = np.hstack((pts_dst, np.ones(shape=(n_pts, 1), dtype=np.float32)))
    # trying all combinations (some of them are invalid)
    for (p0, p1), p2 in iter_product(iter_product(triangle_dst, triangle_dst), triangle_dst):
        N = find_affine(np.vstack((p0, p1, p2)))
        # determinant of affine mapping has to be regular
        if np.linalg.matrix_rank(N) == 3:
            P = np.matmul(N, invM)
            pts_src_mapped = np.matmul(P, pts_src_ext.T).T
            valid, mapping = is_affine_valid(pts_src_mapped, pts_dst_ext)
            if valid:
                return P, mapping
    return None, None

def stars_mapping(P, stars):
    '''
        Applying affine mapping to stars
    '''
    stars = np.flip(stars, axis=1)
    stars = np.hstack((stars, np.ones(shape=(3, 1), dtype=np.float32)))
    cords = np.matmul(P, stars.T).T
    cords = np.flip(np.delete(cords, -1, axis=1), axis=1)
    cords = np.intp(np.round(cords.ravel()))
    return cords

def extreme_triangles(pts):
    '''
        Finding all triangles with vertices on convex hull
        Triangle with biggest area will be among them
    '''
    ch = ConvexHull(pts)
    ch_idx = np.asarray(ch.vertices, dtype=np.int)
    ch_pts = np.asarray([pts[i] for i in ch_idx], dtype=np.float32)
    n_chpts = ch_pts.shape[0]
    triangles = []
    for i in range(n_chpts):
        for j in range(i + 1, n_chpts):
            for k in range(j + 1, n_chpts):
                q0 = ch_pts[i]; q1 = ch_pts[j]; q2 = ch_pts[k]
                area = np.abs(np.cross(q2 - q0, q1 - q0))
                if area > 0:
                    triangles.append((area, q0, q1, q2))
    triangles = np.asarray([*triangles])
    # sorting triangles by area descending
    indices = triangles[:, 0].argsort()
    indices = indices[: : -1]
    triangles = triangles[indices]
    return triangles

def print_array(arr):
    tmp_list = list(arr)
    print(*tmp_list, sep=' ')
    # print(*tmp_list, sep=' ', file=logger)

def find_size_mapping(size_src, size_dst):
    '''
        Aside from barycenters, shapes have to map as well
        For every distorted shape finding in which source shape it maps
    '''
    n_shapes = size_src.shape[0]
    size_mapping = np.zeros(shape=(n_shapes, ), dtype=np.int32)
    size_src_sorted = np.sort(size_src)
    indices = np.argsort(size_dst)
    for i in range(n_shapes):
        index = indices[i]
        size_mapping[index] = size_src_sorted[i]
    return size_mapping

if __name__ == "__main__":
    # with open("log.txt", "w") as logger:
    # barycenter feels smarter
    barycenters_src, size_src, shapes_src = process_src_image()
    barycenters_dst, size_dst = process_dst_image()
    size_mapping = find_size_mapping(size_src, size_dst)
    print_array(np.flip(np.int0(np.round(barycenters_src)), axis=1).ravel())
    print_array(np.flip(np.int0(np.round(barycenters_dst)), axis=1).ravel())

    triangle_src = extreme_triangles(barycenters_src)
    triangle_src = np.asarray([triangle_src[0][1], triangle_src[0][2], triangle_src[0][3]], dtype=np.float32)
    triangle_dst = extreme_triangles(barycenters_dst)
    triangle_dst = np.asarray([[tr[1], tr[2], tr[3]] for tr in triangle_dst], dtype=np.float32)

    invM = np.linalg.inv(find_affine(triangle_src))
    P = None; mapping = None
    for triangle in triangle_dst:
        P, mapping = affine_tryout(invM, barycenters_src, barycenters_dst, triangle)
        if P is not None: break
    shapes_dst = [shapes_src[i] for i in mapping]
    print_array(shapes_src)
    print_array(shapes_dst)

    stars = np.zeros(shape=(3, 2))
    for i in range(3):
        star = input()
        star = star.split(' ')
        stars[i] = np.asarray([star])
    stars = stars_mapping(P, stars)
    print_array(stars)