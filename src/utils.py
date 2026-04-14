"""
utils.py
--------
Funções auxiliares: métricas, visualizações e utilitários gerais.

Paradigma Conexionista — Equipe 4 — Fatec Osasco
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """
    Calcula precision, recall e F1-score macro para classificação multiclasse.

    Retorna dicionário com valores por classe e médias macro.
    """
    precision, recall, f1 = [], [], []

    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 40)
    print(f"  Precision (macro): {metrics['macro_precision']*100:.2f}%")
    print(f"  Recall    (macro): {metrics['macro_recall']*100:.2f}%")
    print(f"  F1-Score  (macro): {metrics['macro_f1']*100:.2f}%")
    print("=" * 40 + "\n")


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def visualize_weights(W: np.ndarray, n_show: int = 16, title: str = "Pesos W1 (features aprendidas)"):
    """
    Visualiza os primeiros n_show filtros da camada W1 como imagens 48×48.
    Útil para interpretar o que a rede aprendeu.
    """
    n_show = min(n_show, W.shape[1])
    cols = min(8, n_show)
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    side = int(np.sqrt(W.shape[0]))  # assume imagem quadrada

    for i in range(n_show):
        w = W[:, i].reshape(side, side)
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
        axes[i].imshow(w_norm, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"N{i+1}", fontsize=7)

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


def visualize_sample_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list,
    n_show: int = 8,
    img_size: tuple = (48, 48),
):
    """
    Exibe amostras com predição e rótulo real lado a lado.
    Borda verde = acerto, vermelha = erro.
    """
    idx = np.random.choice(len(X), n_show, replace=False)
    probs = model.predict_proba(X[idx])
    preds = np.argmax(probs, axis=1)

    cols = min(8, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for j, (i, pred) in enumerate(zip(idx, preds)):
        ax = axes[j]
        img = X[i].reshape(img_size)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        correct = pred == y[i]
        color = "#1D9E75" if correct else "#D85A30"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax.set_title(
            f"Real: {class_names[y[i]]}\nPred: {class_names[pred]}\n{probs[j][pred]*100:.0f}%",
            fontsize=7, color=color
        )

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Predições do modelo MLP", fontsize=11)
    plt.tight_layout()
    return fig


def plot_activation_heatmap(model, X_sample: np.ndarray, layer: str = "h1"):
    """
    Heatmap das ativações de uma camada oculta para um batch de amostras.
    """
    cache = model.forward(X_sample)
    activations = cache[layer]  # (N, H)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(activations.T, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Amostra")
    ax.set_ylabel(f"Neurônio — {layer}")
    ax.set_title(f"Ativações da camada {layer}")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Converte rótulos inteiros em matriz one-hot."""
    oh = np.zeros((y.shape[0], n_classes))
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh
