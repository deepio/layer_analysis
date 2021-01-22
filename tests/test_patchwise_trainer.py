import cProfile

import cv2
import numpy as np
import pytest

from layer_analysis.patchwise.trainer import train_msae


@pytest.mark.slow
def test_training():
  # Read images
  input_image         = cv2.imread("./tests/img/original2.png", cv2.IMREAD_COLOR)
  # Files created by Pixel.js
  background_image    = cv2.imread("./tests/img/background.png", cv2.IMREAD_UNCHANGED)
  selected_reg_image  = cv2.imread("./tests/img/region.png", cv2.IMREAD_UNCHANGED)
  layer1              = cv2.imread("./tests/img/layer1.png", cv2.IMREAD_UNCHANGED)
  layer2              = cv2.imread("./tests/img/layer2.png", cv2.IMREAD_UNCHANGED)
  layer3              = cv2.imread("./tests/img/layer3.png", cv2.IMREAD_UNCHANGED)

  # Create GroundTruth
  regions_mask = (selected_reg_image[:, :, 3] == 255)
  layer1_mask = (layer1[:, :, 3] == 255)
  layer2_mask = (layer2[:, :, 3] == 255)
  layer3_mask = (layer3[:, :, 3] == 255)

  ground_truth = {
    "background": (background_image[:, :, 3] == 255),
    "layer_1": np.logical_and(layer1_mask, regions_mask),
    "layer_2": np.logical_and(layer2_mask, regions_mask),
    "layer_3": np.logical_and(layer3_mask, regions_mask),
  }
  output_path = {
    "background": "./background.hdf5",
    "layer_1": "./layer1.hdf5",
    "layer_2": "./layer2.hdf5",
    "layer_3": "./layer3.hdf5",
  }

  status = train_msae(
    input_image=input_image,
    gt=ground_truth,
    patch_height=16,
    patch_width=16,
    output_path=output_path,
    epochs=1,
    max_samples_per_class=50,
    batch_size=1
  )
  assert True
