"""Unit tests for the feature extractors.  Run from the project root:  pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fractal import FRACTAL_NAMES, box_count_dimension, fractal_features, lacunarity  # noqa: E402
from topology import topological_feature_names, topological_features  # noqa: E402

RNG = np.random.default_rng(0)


def test_topological_vector_length_and_finite():
    f = topological_features(RNG.random((64, 64)).astype(np.float32))
    assert f.shape == (len(topological_feature_names()),)
    assert np.isfinite(f).all()


def test_constant_image_has_one_component():
    f = topological_features(np.full((32, 32), 0.5, dtype=np.float32))
    names = topological_feature_names()
    assert f[names.index("sub_H0_n")] == 1      # a flat image is a single connected component
    assert f[names.index("sub_H1_n")] == 0      # and has no loops


def test_noise_has_more_structure_than_flat():
    names = topological_feature_names()
    noisy = topological_features(RNG.random((64, 64)).astype(np.float32))
    assert noisy[names.index("sub_H0_n")] > 10


def test_box_dimension_of_filled_square_is_two():
    assert abs(box_count_dimension(np.ones((128, 128), dtype=bool)) - 2.0) < 0.05


def test_box_dimension_of_line_is_one():
    img = np.zeros((128, 128), dtype=bool)
    img[64, :] = True
    assert abs(box_count_dimension(img) - 1.0) < 0.1


def test_lacunarity_of_uniform_image_is_one():
    assert abs(lacunarity(np.ones((64, 64), dtype=np.float32), 8) - 1.0) < 1e-6


def test_fractal_vector_matches_names():
    f = fractal_features(RNG.random((128, 128)).astype(np.float32))
    assert f.shape == (len(FRACTAL_NAMES),) and np.isfinite(f).all()
