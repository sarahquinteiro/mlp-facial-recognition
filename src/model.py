"""
model.py
--------
Implementação da rede MLP (Multi Layer Perceptron) com NumPy puro.
Treinamento via Backpropagation com gradiente descendente.

Paradigma Conexionista — Equipe 4 — Fatec Osasco
Ref.: Mitchell (1997); slides Prof.ª Claudia R. Carvalho
"""

import numpy as np
import pickle


# ---------------------------------------------------------------------------
# Funções de ativação
# ---------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit — ativação das camadas ocultas."""
    return np.maximum(0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    """Derivada da ReLU em relação a z (pré-ativação)."""
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax estável numericamente — ativação da camada de saída.
    Converte logits em distribuição de probabilidade.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class MLP:
    """
    Rede Neural Multi Layer Perceptron para reconhecimento facial.

    Arquitetura:
        Entrada (n_features) → Oculta 1 (ReLU) → Oculta 2 (ReLU) → Saída (Softmax)

    Parâmetros
    ----------
    n_input : int
        Número de entradas (pixels achatados), ex: 48*48 = 2304
    hidden1 : int
        Neurônios na 1ª camada oculta (padrão: 128)
    hidden2 : int
        Neurônios na 2ª camada oculta (padrão: 64)
    n_classes : int
        Número de classes (pessoas a identificar)
    lr : float
        Taxa de aprendizado (learning rate)
    optimizer : str
        'sgd' ou 'adam'
    random_state : int
        Semente para reprodutibilidade
    """

    def __init__(
        self,
        n_input: int = 2304,
        hidden1: int = 128,
        hidden2: int = 64,
        n_classes: int = 40,
        lr: float = 0.001,
        optimizer: str = "adam",
        random_state: int = 42,
    ):
        self.n_input = n_input
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.n_classes = n_classes
        self.lr = lr
        self.optimizer = optimizer.lower()
        self.random_state = random_state

        self._init_weights()
        if self.optimizer == "adam":
            self._init_adam()

        # Histórico de treinamento
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def _init_weights(self):
        """He initialization — recomendada para redes com ReLU."""
        rng = np.random.default_rng(self.random_state)

        self.W1 = rng.standard_normal((self.n_input, self.hidden1)) * np.sqrt(2.0 / self.n_input)
        self.b1 = np.zeros((1, self.hidden1))

        self.W2 = rng.standard_normal((self.hidden1, self.hidden2)) * np.sqrt(2.0 / self.hidden1)
        self.b2 = np.zeros((1, self.hidden2))

        self.W3 = rng.standard_normal((self.hidden2, self.n_classes)) * np.sqrt(2.0 / self.hidden2)
        self.b3 = np.zeros((1, self.n_classes))

    def _init_adam(self):
        """Momentos de 1ª e 2ª ordem para o otimizador Adam."""
        zeros = lambda shape: np.zeros(shape)
        self.m = {k: zeros(v.shape) for k, v in self._params().items()}
        self.v = {k: zeros(v.shape) for k, v in self._params().items()}
        self.t = 0   # passo de tempo Adam

    def _params(self) -> dict:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    def count_params(self) -> int:
        return sum(p.size for p in self._params().values())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> dict:
        """
        Propaga X pela rede e retorna ativações de cada camada.

        Retorna dict com z1, h1, z2, h2, z3, output para uso no backward.
        """
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)

        z2 = h1 @ self.W2 + self.b2
        h2 = relu(z2)

        z3 = h2 @ self.W3 + self.b3
        output = softmax(z3)

        return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "z3": z3, "output": output}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def cross_entropy_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-entropy loss com suavização numérica (clip para evitar log(0)).
        output : probabilidades softmax  (N, C)
        y      : rótulos inteiros        (N,)
        """
        n = y.shape[0]
        probs = np.clip(output[np.arange(n), y], 1e-12, 1.0)
        return -np.mean(np.log(probs))

    # ------------------------------------------------------------------
    # Backward pass (Backpropagation)
    # ------------------------------------------------------------------

    def backward(self, X: np.ndarray, y: np.ndarray, cache: dict) -> dict:
        """
        Calcula os gradientes de todos os parâmetros pela regra da cadeia.

        Fluxo:
            Loss → δ_out → δ_h2 → δ_h1 → gradientes de W e b
        """
        n = X.shape[0]
        h1, h2, output = cache["h1"], cache["h2"], cache["output"]
        z1, z2 = cache["z1"], cache["z2"]

        # Gradiente na camada de saída (softmax + cross-entropy → δ = ŷ - y_onehot)
        y_onehot = np.zeros_like(output)
        y_onehot[np.arange(n), y] = 1.0
        delta3 = (output - y_onehot) / n          # (N, C)

        dW3 = h2.T @ delta3
        db3 = delta3.sum(axis=0, keepdims=True)

        # Gradiente na camada oculta 2
        delta2 = (delta3 @ self.W3.T) * relu_deriv(z2)   # (N, H2)

        dW2 = h1.T @ delta2
        db2 = delta2.sum(axis=0, keepdims=True)

        # Gradiente na camada oculta 1
        delta1 = (delta2 @ self.W2.T) * relu_deriv(z1)   # (N, H1)

        dW1 = X.T @ delta1
        db1 = delta1.sum(axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    # ------------------------------------------------------------------
    # Otimização
    # ------------------------------------------------------------------

    def _update_sgd(self, grads: dict):
        """Gradiente descendente simples."""
        for name, param in self._params().items():
            param -= self.lr * grads[name]

    def _update_adam(self, grads: dict, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer (Kingma & Ba, 2015)."""
        self.t += 1
        for name, param in self._params().items():
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * g ** 2
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def _update(self, grads: dict):
        if self.optimizer == "adam":
            self._update_adam(grads)
        else:
            self._update_sgd(grads)

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna vetor de probabilidades por classe."""
        cache = self.forward(X)
        return cache["output"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna a classe com maior probabilidade (argmax)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Salva os pesos da rede em arquivo .pkl."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"Modelo salvo em {path}")

    @classmethod
    def load(cls, path: str) -> "MLP":
        """Carrega pesos de um arquivo .pkl."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls.__new__(cls)
        model.__dict__.update(state)
        print(f"Modelo carregado de {path}")
        return model

    def __repr__(self):
        return (
            f"MLP(input={self.n_input}, hidden1={self.hidden1}, "
            f"hidden2={self.hidden2}, classes={self.n_classes}, "
            f"lr={self.lr}, optimizer={self.optimizer}, "
            f"params={self.count_params():,})"
        )
