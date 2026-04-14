"""
train.py
--------
Loop de treinamento da rede MLP com Backpropagation.
Gera curvas de aprendizado e salva o modelo treinado.

Paradigma Conexionista — Equipe 4 — Fatec Osasco
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import MLP
from preprocessing import load_olivetti, load_dataset_from_dir


# ---------------------------------------------------------------------------
# Loop de treinamento
# ---------------------------------------------------------------------------

def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: bool = True,
):
    """
    Executa o treinamento completo com mini-batches e validação por época.

    A cada época:
      1. Embaralha os dados de treino
      2. Divide em mini-batches
      3. Forward pass → cálculo da loss → Backward pass → atualização de pesos
      4. Avalia no conjunto de validação
    """
    n = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Embaralhar a cada época
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch
        for start in range(0, n, batch_size):
            Xb = X_shuf[start: start + batch_size]
            yb = y_shuf[start: start + batch_size]

            # ---- Forward ----
            cache = model.forward(Xb)

            # ---- Loss ----
            loss = model.cross_entropy_loss(cache["output"], yb)
            epoch_loss += loss
            n_batches += 1

            # ---- Backward ----
            grads = model.backward(Xb, yb, cache)

            # ---- Atualização dos pesos ----
            model._update(grads)

        avg_loss = epoch_loss / n_batches
        train_acc = model.accuracy(X_train, y_train) * 100

        # Validação
        val_cache = model.forward(X_val)
        val_loss = model.cross_entropy_loss(val_cache["output"], y_val)
        val_acc = model.accuracy(X_val, y_val) * 100

        # Histórico
        model.history["loss"].append(avg_loss)
        model.history["acc"].append(train_acc)
        model.history["val_loss"].append(val_loss)
        model.history["val_acc"].append(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Época {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc treino: {train_acc:.1f}% | "
                f"Acc val: {val_acc:.1f}%"
            )

    return model


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_history(history: dict, save_path: str = None):
    """Plota as curvas de loss e acurácia ao longo das épocas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], label="Treino", color="#7F77DD")
    axes[0].plot(epochs, history["val_loss"], label="Validação", color="#1D9E75", linestyle="--")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["acc"], label="Treino", color="#7F77DD")
    axes[1].plot(epochs, history["val_acc"], label="Validação", color="#1D9E75", linestyle="--")
    axes[1].set_title("Acurácia (%)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(model: MLP, X_test: np.ndarray, y_test: np.ndarray,
                          class_names: list, save_path: str = None):
    """Matriz de confusão normalizada."""
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão (normalizada)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Matriz de confusão salva em {save_path}")
    else:
        plt.show()

    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Treinamento MLP — Reconhecimento Facial")
    parser.add_argument("--hidden1", type=int, default=128, help="Neurônios na camada oculta 1")
    parser.add_argument("--hidden2", type=int, default=64,  help="Neurônios na camada oculta 2")
    parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizado")
    parser.add_argument("--epochs", type=int, default=50,   help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do mini-batch")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--data-dir", default=None, help="Diretório com dataset personalizado")
    parser.add_argument("--save-model", default="models/mlp_weights.pkl")
    parser.add_argument("--save-plot", default="docs/curvas_treinamento.png")
    parser.add_argument("--save-cm", default="docs/matriz_confusao.png")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Dados ---
    if args.data_dir:
        X, y, class_names = load_dataset_from_dir(args.data_dir)
    else:
        X, y, class_names = load_olivetti()

    print(f"\nDataset: {X.shape[0]} amostras | {len(class_names)} classes | {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=args.seed, stratify=y_train
    )

    print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

    # --- Modelo ---
    model = MLP(
        n_input=X.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        n_classes=len(class_names),
        lr=args.lr,
        optimizer=args.optimizer,
        random_state=args.seed,
    )
    print(f"\n{model}\n")

    # --- Treinamento ---
    print("=" * 60)
    print("Iniciando treinamento com Backpropagation...")
    print("=" * 60)

    model = train(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # --- Avaliação final ---
    test_acc = model.accuracy(X_test, y_test) * 100
    print(f"\nAcurácia final no conjunto de teste: {test_acc:.2f}%")

    # --- Salvar ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    model.save(args.save_model)

    plot_history(model.history, save_path=args.save_plot)
    plot_confusion_matrix(model, X_test, y_test, class_names, save_path=args.save_cm)


if __name__ == "__main__":
    main()
