# 🧠 Reconhecimento Facial com MLP (Multi Layer Perceptron)

> **Disciplina:** Aprendizagem de Máquina — Fatec Osasco  
> **Equipe 4 — Paradigma Conexionista**  
> **Prof.ª Claudia Rodrigues de Carvalho**

Sistema de reconhecimento facial para controle de acesso utilizando Redes Neurais Artificiais com arquitetura MLP e treinamento via Backpropagation.

---

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura da Rede](#arquitetura-da-rede)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Dataset](#dataset)
- [Resultados](#resultados)
- [Backpropagation](#backpropagation)
- [Equipe](#equipe)

---

## 📌 Sobre o Projeto

Este projeto implementa um sistema de **reconhecimento facial para controle de acesso** utilizando o paradigma **Conexionista** de aprendizado de máquina.

A rede neural artificial (RNA) é inspirada na estrutura do cérebro humano: neurônios interconectados aprendem a identificar rostos através do ajuste iterativo de pesos sinápticos, minimizando o erro de saída pelo método de **Backpropagation**.

### Funcionalidades

- Pré-processamento e normalização de imagens faciais
- Treinamento de rede MLP com múltiplas camadas ocultas
- Backpropagation com gradiente descendente (SGD / Adam)
- Avaliação com métricas de acurácia, precision, recall e F1-score
- Inferência em tempo real via webcam ou imagem estática
- Visualização das curvas de treinamento e ativações

---

## 🏗️ Arquitetura da Rede

```
Entrada        Oculta 1       Oculta 2       Saída
(2304 px)  →  (128 neur.)  →  (64 neur.)  →  (N classes)
  flatten       ReLU            ReLU          Softmax
```

| Camada    | Neurônios | Ativação | Parâmetros  |
|-----------|-----------|----------|-------------|
| Entrada   | 2304      | —        | —           |
| Oculta 1  | 128       | ReLU     | 295.040     |
| Oculta 2  | 64        | ReLU     | 8.256       |
| Saída     | N classes | Softmax  | 64·N + N    |

> Imagens redimensionadas para **48×48 pixels** em escala de cinza → vetor de 2304 entradas.

---

## 📁 Estrutura do Repositório

```
mlp-facial-recognition/
├── src/
│   ├── model.py           # Arquitetura MLP (NumPy puro + scikit-learn)
│   ├── train.py           # Loop de treinamento com backpropagation
│   ├── inference.py       # Inferência em imagem ou webcam
│   ├── preprocessing.py   # Pré-processamento das imagens
│   └── utils.py           # Funções auxiliares e métricas
├── data/
│   ├── raw/               # Imagens originais (não versionadas)
│   └── processed/         # Imagens pré-processadas (.npy)
├── models/
│   └── mlp_weights.pkl    # Pesos salvos após treinamento
├── notebooks/
│   └── exploracao.ipynb   # Análise exploratória e visualizações
├── tests/
│   ├── test_model.py      # Testes unitários da rede
│   └── test_preprocessing.py
├── docs/
│   └── arquitetura.md     # Documentação técnica detalhada
├── demo.py                # Demo rápido com imagens de exemplo
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.9+
- pip

Dependências principais:

```
numpy>=1.24
scikit-learn>=1.3
opencv-python>=4.8
matplotlib>=3.7
pillow>=10.0
tqdm>=4.65
```

---

## 🚀 Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/mlp-facial-recognition.git
cd mlp-facial-recognition

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 💻 Como Usar

### Treinamento

```bash
# Treinamento com configurações padrão
python src/train.py

# Personalizando hiperparâmetros
python src/train.py \
  --hidden1 128 \
  --hidden2 64 \
  --lr 0.001 \
  --epochs 50 \
  --batch-size 32 \
  --optimizer adam
```

### Inferência em imagem estática

```bash
python src/inference.py --image caminho/para/foto.jpg
```

### Demo rápido

```bash
python demo.py
```

### Notebook exploratório

```bash
jupyter notebook notebooks/exploracao.ipynb
```

---

## 🗃️ Dataset

O projeto utiliza o dataset **Olivetti Faces** (AT&T), disponível via `scikit-learn`:

- 400 imagens de 40 pessoas (10 por pessoa)
- Resolução original: 64×64 pixels em escala de cinza
- Pré-processamento aplicado: redimensionamento → normalização [0,1] → flatten

```python
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
```

Para usar **seu próprio dataset**, organize as imagens assim:

```
data/raw/
├── pessoa_01/
│   ├── foto1.jpg
│   └── foto2.jpg
├── pessoa_02/
│   └── ...
```

E execute o pré-processamento:

```bash
python src/preprocessing.py --input data/raw --output data/processed
```

---

## 📊 Resultados

Resultados obtidos no dataset Olivetti Faces (80% treino / 20% teste):

| Métrica       | Valor  |
|---------------|--------|
| Acurácia      | ~95%   |
| Precision     | ~94%   |
| Recall        | ~94%   |
| F1-Score      | ~94%   |
| Tempo treino  | ~8s    |

> Resultados podem variar conforme hiperparâmetros e semente aleatória.

---

## 🔁 Backpropagation

O algoritmo de backpropagation ajusta os pesos da rede em 4 etapas por época:

1. **Forward pass** — propaga a entrada até a saída calculando ativações
2. **Cálculo do erro** — cross-entropy loss entre predição e rótulo real
3. **Backward pass** — propaga o gradiente do erro de volta pela rede (regra da cadeia)
4. **Atualização dos pesos** — `w = w - α * ∂L/∂w`

```python
# Gradiente camada de saída
delta_out = softmax_output - y_onehot

# Gradiente camada oculta 2
delta_h2 = (delta_out @ W3.T) * relu_deriv(z2)

# Gradiente camada oculta 1
delta_h1 = (delta_h2 @ W2.T) * relu_deriv(z1)

# Atualização
W3 -= lr * h2.T @ delta_out
W2 -= lr * h1.T @ delta_h2
W1 -= lr * X.T  @ delta_h1
```

---

## 👥 Equipe

| Nome | RA |
|------|----|
| Karine Fernandes e Silva | 2161392413030 |
| Sarah Quinteiro Limeira da Silva | 2161392413027 |

**Fatec Osasco — Análise e Desenvolvimento de Sistemas**  
Disciplina: Aprendizagem de Máquina — 2026

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
