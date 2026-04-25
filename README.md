# Digit Recognizer

Solução para a competição [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) do Kaggle.

---

## Resultado

| Métrica | Valor |
|---|---|
| Acurácia (Kaggle) | **98.089%** |
| Algoritmo | MLP — Multilayer Perceptron |
| Dataset | MNIST (42.000 imagens de treino) |

---

## Estrutura

```
digit-recognizer-kaggle/
├── digit_recognizer_julianaburato.ipynb
├── README.md
└── submission.csv
```

---

## Etapas do Projeto

### 1. Análise Exploratória (EDA)

- Distribuição das classes — dataset balanceado (~4.200 exemplos por dígito)
- Visualização das imagens para identificar variações de escrita e possíveis confusões entre classes (ex: 4↔9, 3↔8)
- Análise dos valores de pixel (intervalo 0–255)

### 2. Pré-processamento

- Normalização dos pixels para [0.0, 1.0] — gradientes mais estáveis durante o backpropagation
- Reshape de `(n, 784)` para `(n, 28, 28, 1)` — preserva a estrutura espacial da imagem
- One-hot encoding dos labels — formato exigido pela loss `categorical_crossentropy`
- Divisão 80/20 treino/validação

### 3. Data Augmentation

Aplicada via `ImageDataGenerator` para melhorar a generalização. Transformações conservadoras — espelhamento e rotações >15° alterariam a identidade do dígito:

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `rotation_range` | 10° | Inclinação natural da escrita |
| `zoom_range` | 10% | Variação de tamanho entre escritores |
| `width_shift_range` | 10% | Deslocamento horizontal |
| `height_shift_range` | 10% | Deslocamento vertical |

### 4. Modelo: MLP

#### Arquitetura

Rede neural densa com 3 camadas ocultas, dropout decrescente e BatchNormalization em cada camada:

```
Input (28×28×1) → Flatten (784)
    → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    → Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
    → Dense(10)  + Softmax
```

| Componente | Função |
|---|---|
| BatchNormalization | Estabiliza e acelera o treino normalizando as ativações |
| Dropout | Regularização — desativa neurônios aleatoriamente para evitar overfitting |
| Softmax | Converte logits em distribuição de probabilidade (soma = 1) |

#### Treinamento

- Optimizer: Adam — adapta o learning rate por parâmetro
- Loss: Categorical Crossentropy
- EarlyStopping (`patience=10`, `restore_best_weights=True`)
- ReduceLROnPlateau (`factor=0.5`, `patience=5`)

### 5. Avaliação

- Acurácia no Kaggle: **98.089%**
- Matriz de confusão e relatório de classificação por dígito
- Visualização dos exemplos onde o modelo errou

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

*Juliana Burato — 2026*
