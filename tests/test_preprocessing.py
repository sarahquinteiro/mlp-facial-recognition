"""
tests/test_preprocessing.py
----------------------------
Testes unitários do módulo de pré-processamento.
Execute: python -m pytest tests/ -v

Equipe 4 — Fatec Osasco
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from PIL import Image

from preprocessing import normalize, flatten, load_image, TARGET_SIZE


class TestNormalize:
    def test_range(self):
        img = np.array([[0, 128, 255]], dtype=np.float32)
        norm = normalize(img)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_zeros(self):
        img = np.zeros((48, 48), dtype=np.float32)
        assert np.allclose(normalize(img), 0)

    def test_ones(self):
        img = np.full((48, 48), 255, dtype=np.float32)
        assert np.allclose(normalize(img), 1.0)


class TestFlatten:
    def test_shape(self):
        img = np.ones((48, 48))
        vec = flatten(img)
        assert vec.shape == (2304,)

    def test_values_preserved(self):
        img = np.arange(48 * 48, dtype=float).reshape(48, 48)
        vec = flatten(img)
        assert np.allclose(vec, img.flatten())


class TestLoadImage:
    def test_target_size(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        img = Image.fromarray(np.random.randint(0, 255, (100, 80), dtype=np.uint8))
        img.save(img_path)
        loaded = load_image(img_path)
        assert loaded.shape == TARGET_SIZE

    def test_grayscale(self, tmp_path):
        img_path = str(tmp_path / "color.png")
        rgb = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        rgb.save(img_path)
        loaded = load_image(img_path)
        assert loaded.ndim == 2
