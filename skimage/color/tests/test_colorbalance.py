# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:57:01 2014

@author: chuong
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for color balance functions.

Authors
-------
- the rgb2hsv test was written by Chuong Nguyen, 2014

:license: modified BSD
"""

import os.path

import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           assert_raises,
                           TestCase,
                           )

from skimage import img_as_float, img_as_ubyte
from skimage.io import imread
from skimage import data_dir, data
from skimage.color import (get_colorcard_colors,
                           get_color_correction_parameters,
                           correct_color)


class TestColorbalance(TestCase):
    img_rgb = imread(os.path.join(data_dir, 'cropped_color_card.png'))
    color_card = img_rgb[743:829, 1284:1415]
    actual_colors = get_colorcard_colors(color_card, grid_size=[6, 4])
    true_color_card = 255.0*imread(os.path.join(data_dir,
                                   'CameraTrax_24ColorCard_2x3in.png'))
    true_colors = get_colorcard_colors(true_color_card, grid_size=[6, 4])


    color_alpha_array = np.array([[0.98844109, 0.1113885, -0.13325764],
                                  [0.07231765, 0.66030949, 0.03172648],
                                  [-0.09717971, 0.25484042, 0.76104714]])
    color_constant_array = np.array([[-11.22658842],
                                     [43.58277321],
                                     [-8.24595865]])
    color_gamma_array = np.array([[2.67669598],
                                  [1.90837229],
                                  [2.02217615]])

    def test_get_color_correction_parameters(self):
        self.color_alpha, self.color_constant, self.color_gamma = \
            get_color_correction_parameters(self.true_colors,
                                            self.actual_colors)
        assert_array_almost_equal(self.color_alpha, self.color_alpha_array)
        assert_array_almost_equal(self.color_constant,
                                  self.color_constant_array)
        assert_array_almost_equal(self.color_gamma, self.color_gamma_array)

    def test_correct_color(self):
        self.corrected_color_card = \
            correct_color(self.color_card, self.color_alpha,
                          self.color_constant, self.color_gamma)
        assert_array_almost_equal(self.corrected_color_card,
                                  self.true_color_card)

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
