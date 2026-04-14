"""
tests/test_model.py
-------------------
Testes unitários da rede MLP.
Execute: python -m pytest tests/ -v

Equipe 4 — Fatec Osasco
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from model import MLP, relu, relu_deriv, softmax


# ---------------------------------------------------------------------------
# Funções de ativação
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu_positive(self):
        x = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(relu(x), x)

    def test_relu_negative(self):
        x = np.array([[-1.0, -2.0, 0.0]])
        expected = np.array([[0.0, 0.0, 0.0]])
        assert np.allclose(relu(x), expected)

    def test_relu_deriv_shape(self):
        x = np.random.randn(4, 10)
        assert relu_deriv(x).shape == x.shape

    def test_softmax_sums_to_one(self):
        z = np.random.randn(5, 8)
        probs = softmax(z)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_softmax_non_negative(self):
        z = np.random.randn(3, 5)
        assert np.all(softmax(z) >= 0)

    def test_softmax_numerical_stability(self):
        z = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(z)
        assert not np.any(np.isnan(probs))
        assert np.allclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# Inicialização do modelo
# ---------------------------------------------------------------------------

class TestMLPInit:
    def setup_method(self):
        self.model = MLP(n_input=100, hidden1=32, hidden2=16, n_classes=5, lr=0.01)

    def test_weight_shapes(self):
        assert self.model.W1.shape == (100, 32)
        assert self.model.W2.shape == (32, 16)
        assert self.model.W3.shape == (16, 5)

    def test_bias_shapes(self):
        assert self.model.b1.shape == (1, 32)
        assert self.model.b2.shape == (1, 16)
        assert self.model.b3.shape == (1, 5)

    def test_param_count(self):
        # W1: 100*32 + 32, W2: 32*16 + 16, W3: 16*5 + 5
        expected = (100*32+32) + (32*16+16) + (16*5+5)
        assert self.model.count_params() == expected


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForward:
    def setup_method(self):
        self.model = MLP(n_input=50, hidden1=20, hidden2=10, n_classes=4)
        self.X = np.random.rand(8, 50)

    def test_output_shape(self):
        cache = self.model.forward(self.X)
        assert cache["output"].shape == (8, 4)

    def test_output_is_probability(self):
        cache = self.model.forward(self.X)
        assert np.allclose(cache["output"].sum(axis=1), 1.0, atol=1e-5)
        assert np.all(cache["output"] >= 0)

    def test_hidden_shapes(self):
        cache = self.model.forward(self.X)
        assert cache["h1"].shape == (8, 20)
        assert cache["h2"].shape == (8, 10)

    def test_relu_activations_non_negative(self):
        cache = self.model.forward(self.X)
        assert np.all(cache["h1"] >= 0)
        assert np.all(cache["h2"] >= 0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLoss:
    def setup_method(self):
        self.model = MLP(n_input=10, hidden1=8, hidden2=4, n_classes=3)

    def test_loss_positive(self):
        output = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y = np.array([0, 1])
        loss = self.model.cross_entropy_loss(output, y)
        assert loss > 0

    def test_perfect_prediction_low_loss(self):
        output = np.array([[0.99, 0.005, 0.005]])
        y = np.array([0])
        loss = self.model.cross_entropy_loss(output, y)
        assert loss < 0.1

    def test_wrong_prediction_high_loss(self):
        output = np.array([[0.005, 0.005, 0.99]])
        y = np.array([0])
        loss = self.model.cross_entropy_loss(output, y)
        assert loss > 2.0


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

class TestBackward:
    def setup_method(self):
        self.model = MLP(n_input=20, hidden1=10, hidden2=5, n_classes=3, lr=0.01)
        self.X = np.random.rand(4, 20)
        self.y = np.array([0, 1, 2, 0])

    def test_gradient_shapes(self):
        cache = self.model.forward(self.X)
        grads = self.model.backward(self.X, self.y, cache)
        assert grads["W1"].shape == self.model.W1.shape
        assert grads["W2"].shape == self.model.W2.shape
        assert grads["W3"].shape == self.model.W3.shape

    def test_loss_decreases_after_update(self):
        cache = self.model.forward(self.X)
        loss_before = self.model.cross_entropy_loss(cache["output"], self.y)

        grads = self.model.backward(self.X, self.y, cache)
        self.model._update_sgd(grads)

        cache2 = self.model.forward(self.X)
        loss_after = self.model.cross_entropy_loss(cache2["output"], self.y)

        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def setup_method(self):
        self.model = MLP(n_input=15, hidden1=8, hidden2=4, n_classes=3)
        self.X = np.random.rand(5, 15)

    def test_predict_shape(self):
        preds = self.model.predict(self.X)
        assert preds.shape == (5,)

    def test_predict_valid_classes(self):
        preds = self.model.predict(self.X)
        assert np.all(preds >= 0) and np.all(preds < 3)

    def test_predict_proba_shape(self):
        probs = self.model.predict_proba(self.X)
        assert probs.shape == (5, 3)


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load(self, tmp_path):
        model = MLP(n_input=10, hidden1=6, hidden2=3, n_classes=2)
        X = np.random.rand(3, 10)
        preds_before = model.predict(X)

        path = str(tmp_path / "model.pkl")
        model.save(path)

        loaded = MLP.load(path)
        preds_after = loaded.predict(X)

        assert np.array_equal(preds_before, preds_after)
