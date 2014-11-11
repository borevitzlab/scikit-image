#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for color balance functions.

Authors
-------
- the colorbalance test was written by Chuong Nguyen, 2014

:license: modified BSD
"""

import os.path

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           TestCase,
                           )

from skimage.io import imread
from skimage import data_dir
from skimage.color import (get_colorcard_colors,
                           get_color_correction_parameters,
                           correct_color)


class TestColorbalance(TestCase):
    # input data
    color_card = imread(os.path.join(data_dir, 'cropped_color_card.png'),
                        plugin="freeimage")
    actual_colors = get_colorcard_colors(color_card, grid_size=[6, 4])
    true_color_card = imread(os.path.join(data_dir,
                                          'CameraTrax_24ColorCard_2x3in.png'),
                             plugin="freeimage")
    true_colors = get_colorcard_colors(true_color_card, grid_size=[6, 4])

    # predicted results
    color_alpha = np.array([[0.98609494, 0.1093022, -0.13196349],
                           [0.07417578, 0.65261063, 0.03358806],
                           [-0.09229596, 0.25357092, 0.75431669]])
    color_constant = np.array([[-10.38953662],
                               [45.40771496],
                               [-6.64889069]])
    color_gamma = np.array([[2.69716508],
                           [1.94444779],
                           [2.05671531]])
    corrected_color_card = imread(os.path.join(data_dir,
                                               'corrected_color_card.png'),
                                  plugin="freeimage")

    def test_get_color_correction_parameters(self):
        color_alpha, color_constant, color_gamma = \
            get_color_correction_parameters(self.true_colors,
                                            self.actual_colors)
        assert_array_almost_equal(color_alpha, self.color_alpha)
        assert_array_almost_equal(color_constant, self.color_constant)
        assert_array_almost_equal(color_gamma, self.color_gamma)

    def test_correct_color(self):
        color_alpha, color_constant, color_gamma = \
            get_color_correction_parameters(self.true_colors,
                                            self.actual_colors)
        corrected_color_card = \
            correct_color(self.color_card, self.color_alpha,
                          self.color_constant, self.color_gamma)
        assert_array_almost_equal(corrected_color_card,
                                  self.corrected_color_card)

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
