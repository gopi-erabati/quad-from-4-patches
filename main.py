import argparse
import cv2
import numpy as np


def find_brightest_patch_centers(img, patch_size=5, num_patches=4):
    """
    Finds the four brightest non-overlapping 5x5 patches in an image.

    Args:
        img (ndarray): The input image as a NumPy array.
        patch_size (int): The size of the patches to find.
        num_patches (int): The number of patches to find.

    Returns:
        [(tuple)] where each tuple contains the center of patch.
    """

    height, width = img.shape
    assert patch_size <= height and patch_size <= width

    # patch extraction using strides
    patches = np.lib.stride_tricks.as_strided(
        img,
        shape=(height - patch_size + 1, width - patch_size + 1, patch_size,
               patch_size),
        strides=img.strides + img.strides
    )

    # flatten patches and calculate avg brightness
    flattened_patches = patches.reshape(-1, patch_size * patch_size)
    patch_brightness = np.mean(flattened_patches, axis=1)

    # sort patches by brightness in des. order
    patch_indices = np.argsort(patch_brightness)[::-1]

    # select non-overlapping patches
    selected_patches = []
    for i in patch_indices:
        x = (i // (width - patch_size + 1)) + patch_size // 2
        y = (i % (width - patch_size + 1)) + patch_size // 2
        if not any(abs(x - px) < patch_size and abs(y - py) < patch_size
                   for px, py in selected_patches):
            selected_patches.append((x, y))
            if len(selected_patches) == num_patches:
                break

    if len(selected_patches) < num_patches:
        raise ValueError(f"Can't find {num_patches} non-overlapping "
                         f"{patch_size}x{patch_size} patches, "
                         f"decrease the patch_size or num_patches")

    return selected_patches


def sort_quad_points(pts):
    """
    Sort the quad points anticlockwise

    Args:
        pts (List[tuple]): points to sort

    Returns:
        List[tuple]
    """
    pts = np.array(pts, dtype=np.int32)
    # order points anticlockwise to ensure quad is drawn correctly
    centroid = np.mean(pts, axis=0)

    # function to sort points based on angle from the centroid
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

    # sort the points based on their angle relative to the centroid
    ordered_pts = sorted(pts, key=angle_from_centroid)

    return ordered_pts


def cal_quad_area(points):
    """
    Calculates the area of quadrilateral using shoelace formula for sorted
    anticlockwise points

    Args:
        points (List[tuple]): points to cal area sorted anticlockwise

    Returns:
        (float) The area of quad.
    """
    # Shoelace formula
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area


def draw_quad(image, pts, color=(0, 0, 255)):
    """
    Draw the quadrilateral in red color of points sorted anticlockwise

    Args:
        image (ndarray): input image
        pts (List[tuple]): corners of quad
        color (tuple): color of quad lines

    Returns:
        (ndarrray): Image with drawn quad in red
    """
    pts = [(t[1], t[0]) for t in pts]  # polylines expect (w, h)
    ordered_pts = np.array(pts).reshape((-1, 1, 2))
    cv2.polylines(image, [ordered_pts], isClosed=True, color=color,
                  thickness=2)


def find_brightpatches_draw_quad(img_input, img_output):
    """
    Find the 4 non-overlapping bright patches, cal area of quad, and draw quad
    in red

    Args:
        img_input (ndarray): input image
        img_output (ndarray): output image with drawn quad in red
    """

    # read the input image in grayscale
    img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_input}' not found.")

    # find nonoverlapping bright patches and their centers
    pts = find_brightest_patch_centers(img)
    # [(x1,y1), (x2, y2), (x3, y3), (x4, y4)]

    # calculate the area of the quad
    # first sort the points anticlockwise
    ordered_pts = sort_quad_points(pts)
    area = cal_quad_area(ordered_pts)
    print(f"Area of the quadrilateral: {area} pixels²")

    # convert grayscale image to BGR for drawing colored lines
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw the quad
    draw_quad(img_rgb, ordered_pts)

    # save the result
    cv2.imwrite(img_output, img_rgb)
    print(f'Output image stored at {img_output}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Finds the four non-overlapping 5x5 patches with highest '
                    'average brightness, takes the patch centers as corners '
                    'of a quadrilateral, calculates its area in pixels, '
                    'and draws the quadrilateral in red into the image and '
                    'saves it in PNG format')
    parser.add_argument('--input', type=str,
                        default='./input.jpg', help='Path of input image '
                                                    'filename ('
                                                    'default=\'./input.jpg\')')
    parser.add_argument('--output', type=str,
                        default='./output.png',
                        help='Path to output image filename ('
                             'default=\'./output.png\')')

    args = parser.parse_args()

    find_brightpatches_draw_quad(args.input, args.output)
