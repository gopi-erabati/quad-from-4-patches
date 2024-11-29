import unittest
import numpy as np
import sys
import os
# Add current directory to sys.path
sys.path.append(os.path.abspath('.'))

from main import (find_brightest_patch_centers, cal_quad_area, draw_quad,
                  sort_quad_points)


class TestPatchBrightnessAndQuadrilateral(unittest.TestCase):

    # Test find_brightest_patch_centers() with various cases
    def setUp(self):
        self.test_img = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [1, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [1, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [1, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [1, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ], dtype=np.uint8)

    def test_find_brightest_patch_centers(self):
        """ test to find brightest non-overlapping patches"""
        patches = find_brightest_patch_centers(self.test_img)
        self.assertEqual(len(patches), 4)
        expected = [(7, 7), (2, 7), (7, 2), (2, 2)]
        self.assertTrue(all(x in patches for x in expected))

    def test_find_brightest_patch_centers_1_patch(self):
        result = find_brightest_patch_centers(self.test_img, patch_size=5,
                                              num_patches=1)
        expected = [(7, 7)]
        self.assertEqual(result, expected)

    def test_find_brightest_patch_centers_largerpatch(self):
        with self.assertRaises(AssertionError):
            find_brightest_patch_centers(self.test_img, patch_size=12,
                                         num_patches=1)

    def test_find_brightest_patch_centers_patchsize_1(self):
        patches = find_brightest_patch_centers(self.test_img, patch_size=1,
                                               num_patches=1)
        expected = [(9, 9)]  # Brightest pixel should be (9, 9) with value 9
        self.assertEqual(patches, expected)

    def test_find_brightest_patch_centers_morepatches(self):
        with self.assertRaises(ValueError):
            find_brightest_patch_centers(self.test_img, patch_size=5,
                                         num_patches=10)

    def test_find_brightest_patch_centers_largeimg(self):
        large_img = np.random.randint(0, 255, (128, 128))
        patches = find_brightest_patch_centers(large_img, patch_size=5,
                                               num_patches=4)
        self.assertEqual(len(patches), 4)

    def test_find_brightest_patch_centers_minimgsize(self):
        test_img = np.array([
            [10, 20, 30, 40, 50],
            [50, 40, 30, 20, 10],
            [10, 20, 30, 40, 50],
            [50, 40, 30, 20, 10],
            [10, 20, 30, 40, 50],
        ], dtype=np.uint8)

        patches = find_brightest_patch_centers(test_img, patch_size=5,
                                               num_patches=1)
        expected = [(2, 2)]
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches, expected)

    def test_find_brightest_patch_centers_nonsquare(self):
        test_img = np.array([
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        ], dtype=np.uint8)

        patches = find_brightest_patch_centers(test_img, patch_size=5,
                                               num_patches=2)
        expected = [(2, 7), (2, 2)]
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches, expected)

    def test_find_brightest_patch_centers_imgbrightspots(self):
        test_img = np.zeros((100, 100), dtype=np.uint8)
        test_img[10:15, 10:15] = 255
        test_img[50:55, 50:55] = 250
        test_img[90:95, 90:95] = 245
        test_img[30:35, 70:75] = 240

        patches = find_brightest_patch_centers(test_img, patch_size=5,
                                               num_patches=4)
        expected = [(12, 12), (52, 52), (92, 92), (32, 72)]
        self.assertEqual(len(patches), 4)
        self.assertEqual(patches, expected)

    # Test sort_quad_points()
    def test_sort_quad_points(self):
        points = [(0, 0), (0, 5), (5, 5), (5, 0)]
        sorted_points = sort_quad_points(points)
        expected = [(0, 0), (5, 0), (5, 5), (0, 5)]
        self.assertEqual(np.array(sorted_points).all(),
                         np.array(expected).all())

    # Test cal_quad_area() with various cases
    def test_cal_quad_area_square(self):
        points = [(0, 0), (5, 0), (5, 5), (0, 5)]
        area = cal_quad_area(points)
        expected = 25.0
        self.assertAlmostEqual(area, expected)

    def test_cal_quad_area_rect(self):
        points = [(0, 0), (5, 0), (5, 2), (0, 2)]
        area = cal_quad_area(points)
        expected = 10.0
        self.assertAlmostEqual(area, expected)

    def test_cal_quad_area_irregularquad(self):
        points = [(0, 0), (2, 0), (2, 3), (0, 1)]
        area = cal_quad_area(points)
        expected = 4.0
        self.assertAlmostEqual(area, expected)

    def test_cal_quad_area_zero(self):
        points = [(0, 0), (0, 0), (0, 0), (0, 0)]
        area = cal_quad_area(points)
        expected = 0.0
        self.assertAlmostEqual(area, expected)

    def test_cal_quad_area_collinear(self):
        points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        area = cal_quad_area(points)
        expected = 0.0
        self.assertAlmostEqual(area, expected)

    def test_cal_quad_area_nonconvex(self):
        points = [(100, 100), (300, 50), (250, 250), (150, 200)]
        area = cal_quad_area(points)
        expected = 22500.0
        self.assertAlmostEqual(area, expected)

    # Test draw_quad() with various cases
    def test_draw_quad(self):
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        points = [(25, 25), (75, 25), (75, 75), (25, 75)]
        draw_quad(test_image, points)
        self.assertEqual(test_image[50, 25, 2], 255)  # Checking for red color


if __name__ == '__main__':
    unittest.main()
