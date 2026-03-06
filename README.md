# Construindo o Transformer Encoder "From Scratch" 

Implementação manual de um **Transformer Encoder** em Python puro com NumPy e Pandas, seguindo a arquitetura do paper *"Attention Is All You Need"*. O modelo processa a frase `"Eu tenho apenas dez reais na minha conta"`.

---

## Pré-requisitos

- Python **3.8+**
- pip

---

## Instalação

```bash
# Clone ou baixe (ou copie e cole) o arquivo transformer_encoder.py
# Em seguida, instale as dependências:

pip install numpy pandas
```

---

## Como executar

```bash
python transformer_encoder.py
```

---

## Saída esperada

```
Dimensoes iniciais: (1, 8, 64)
Dimensoes finais:   (1, 8, 64)

Z final (primeiros 8 valores de cada token)
[0] Eu        : [+valor  +valor  ...]
[1] tenho     : [+valor  +valor  ...]
...
[7] conta     : [+valor  +valor  ...]
```

- As **dimensões** devem se manter `(1, 8, 64)` do início ao fim.
- Os **valores de Z** são diferentes de X, pois cada token agora carrega informação contextual dos demais tokens da frase.

---

## Estrutura do código

| Passo | O que faz |
|-------|-----------|
| **Passo 1 — Preparação dos dados** | Constrói vocabulário, tabela de embeddings aleatória e converte a frase em IDs |
| **Passo 2 — Motor matemático** | Implementa `softmax`, `scaled_dot_product_attention`, `MultiHeadAttention`, `layer_norm` e `FeedForwardNetwork` |
| **Passo 3 — Empilhamento** | Empilha `N=6` blocos `EncoderBlock` e passa a entrada por todos eles |

---

## Observações

- Os pesos são sempre random, então os valores numéricos de Z variam entre execuções.
- A multiplicação por `0.1` na inicialização dos pesos da `MultiHeadAttention` evita gradientes muito saturados (by IA).

## Uso de IA
A IA generativa foi usada nos seguintes contextos:
- Sintaxe do python, numpy e pandas
- Estilização desse README.
