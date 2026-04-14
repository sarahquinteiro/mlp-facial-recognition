# Documentação Técnica — Arquitetura MLP

## Paradigma Conexionista

O **paradigma conexionista** modela o aprendizado de forma análoga ao funcionamento do cérebro humano. Unidades de processamento (neurônios artificiais) são interconectadas por sinapses com pesos ajustáveis. O conhecimento emerge do padrão de pesos, não de regras explícitas.

---

## Arquitetura da Rede

### Diagrama de camadas

```
Entrada           Oculta 1         Oculta 2          Saída
┌─────────┐      ┌──────────┐     ┌──────────┐     ┌────────┐
│  2304   │─────▶│   128    │────▶│    64    │────▶│   40   │
│ neurôn. │      │ neurôn.  │     │ neurôn.  │     │ neurôn.│
│         │      │  (ReLU)  │     │  (ReLU)  │     │(Softmx)│
└─────────┘      └──────────┘     └──────────┘     └────────┘
  pixels           aprende          aprende          classe
  da face          bordas           padrões          predita
                   e texturas       compostos
```

### Dimensões das matrizes de pesos

| Parâmetro | Dimensão      | Inicialização   |
|-----------|---------------|-----------------|
| W1        | (2304, 128)   | He (√2/fan_in)  |
| b1        | (1, 128)      | Zeros           |
| W2        | (128, 64)     | He (√2/fan_in)  |
| b2        | (1, 64)       | Zeros           |
| W3        | (64, 40)      | He (√2/fan_in)  |
| b3        | (1, 40)       | Zeros           |

**Total de parâmetros treináveis: ~303.400**

---

## Funções de Ativação

### ReLU (Rectified Linear Unit) — camadas ocultas

```
f(z) = max(0, z)
f'(z) = 1 se z > 0, senão 0
```

Vantagens:
- Evita o problema do vanishing gradient (comparado à sigmoid/tanh)
- Computacionalmente eficiente
- Induz esparsidade nas ativações

Limitação conhecida: **Dying ReLU** — neurônios com z ≤ 0 permanente param de aprender.

### Softmax — camada de saída

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

Garante que a saída seja uma distribuição de probabilidade válida (soma = 1), interpretável como a confiança da rede para cada classe.

---

## Backpropagation

### Fundamento teórico

O algoritmo utiliza a **regra da cadeia** do cálculo diferencial para propagar o gradiente da função de perda de volta pelas camadas, permitindo calcular a contribuição de cada peso no erro final.

### Fluxo completo

```
1. Forward pass
   X → z1=X·W1+b1 → h1=ReLU(z1) → z2=h1·W2+b2 → h2=ReLU(z2) → z3=h2·W3+b3 → ŷ=softmax(z3)

2. Cálculo da loss (cross-entropy)
   L = -Σ y_true · log(ŷ)

3. Gradiente na camada de saída
   δ3 = (ŷ - y_onehot) / N

4. Gradiente W3 e b3
   ∂L/∂W3 = h2ᵀ · δ3
   ∂L/∂b3 = Σ δ3

5. Propagação para camada oculta 2
   δ2 = (δ3 · W3ᵀ) ⊙ ReLU'(z2)

6. Gradiente W2 e b2
   ∂L/∂W2 = h1ᵀ · δ2

7. Propagação para camada oculta 1
   δ1 = (δ2 · W2ᵀ) ⊙ ReLU'(z1)

8. Gradiente W1 e b1
   ∂L/∂W1 = Xᵀ · δ1

9. Atualização dos pesos (SGD)
   W ← W - α · ∂L/∂W
```

---

## Otimizadores

### SGD (Stochastic Gradient Descent)
```
w = w - α · ∂L/∂w
```
Simples, mas pode oscilar e convergir lentamente.

### Adam (Adaptive Moment Estimation)
```
m_t = β1 · m_(t-1) + (1-β1) · g_t       # 1º momento (média)
v_t = β2 · v_(t-1) + (1-β2) · g_t²      # 2º momento (variância)
m̂_t = m_t / (1-β1^t)                    # correção de viés
v̂_t = v_t / (1-β2^t)
w = w - α · m̂_t / (√v̂_t + ε)
```

Padrões: β1=0.9, β2=0.999, ε=1e-8, α=0.001

---

## Pipeline de Pré-processamento

```
Imagem bruta (qualquer tamanho)
         ↓
  Converter para escala de cinza
         ↓
  Redimensionar para 48×48 px
         ↓
  Normalizar pixels: [0, 255] → [0.0, 1.0]
         ↓
  Achatar: (48, 48) → (2304,)
         ↓
  Vetor de entrada da MLP
```

---

## Referências

- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
- Carvalho, C. R. (2024). Slides — Aprendizado Probabilístico e Paradigmas de AM. Fatec Osasco.
