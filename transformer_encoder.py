import numpy as np
import pandas as pd

# PASSO 1: Preparacao dos Dados
# Nesse passo, eu preparei os dados incluindo a construção do vocabulário, os hiperparâmetros,
# a criação da tabela de embeddings
# e a conversão da frase de entrada para IDs. 
# O resultado é uma matriz de embeddings pronta para ser processada pelo encoder.

# Frase usada: "Eu tenho apenas dez reais na minha conta"
vocab = {
    "palavra": ["Eu", "tenho", "apenas", "dez", "reais", "na", "minha", "conta"],
    "id":      [0, 1, 2, 3, 4, 5, 6, 7]
}

# Aqui foi criado um DataFrame para converter as palavras para IDs
df_vocab = pd.DataFrame(vocab).set_index("palavra")

# Depois foi criado um dicionario para mapear palavras para IDs, e aí obter o tamanho do vocabulário
word_to_id = df_vocab["id"].to_dict()
vocab_size  = len(word_to_id)

# Definiu-se a frase de entrada, os IDs correspondentes, e os hiperparametros do modelo
frase = ["Eu", "tenho", "apenas", "dez", "reais", "na", "minha", "conta"]
input_ids = [word_to_id[w] for w in frase]
d_model = 64 # Dimensão do modelo(foi indicado o valor 64)
d_ffn    = 256 # Dimensão da camada feed-forward-network
h       = 8 # Número de cabeças
d_k     = d_model // h # Dimensão de cada cabeça
N       = 6 # Número de blocos do encoder

# Aqui, criei uma tabela de embeddings aleatória para o vocabulário
embedding_table = np.random.randn(vocab_size, d_model)

# Para finalizar essa estruturação, converti os IDs de entrada para suas representações de embedding
# e adicionei uma dimensão de batch (1, seq_len, d_model)
X = embedding_table[input_ids]
X = X[np.newaxis, :, :]
batch_size, seq_len, _ = X.shape



# PASSO 2: Funcoes do Encoder
# Nesse passo, implementei a self attention, a classe de multi-head attention, 
# a função de normalização e a classe de feed-forward.

def softmax(x):
    # Softmax criado "literalmente"
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Self-attention normal
def scaled_dot_product_attention(Q, K, V):
    d_k_local = Q.shape[-1] # Pega o valor da dimensão das cebeças
    scores  = Q @ K.transpose(0, 2, 1) # Calcula os scores (Q * K^T)
    scores  = scores / np.sqrt(d_k_local) # Suaviza os scores dividindo pela raiz quadrada da dimensão das cabeças
    weights = softmax(scores) # Aplica softmax para obter os pesos de atenção
    output  = weights @ V # Calcula a saída ponderada (weights * V)
    return output, weights


class MultiHeadAttention:
    def __init__(self, d_model, h):
        self.h       = h
        self.d_k     = d_model // h
        self.d_model = d_model

        # Pesos por cabeca:
        self.W_Q = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_K = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_V = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]

        # Pesos globais que recebem concatenação de todas as cabecas
        self.W_Q_G = np.random.randn(d_model, d_model) * 0.1
        self.W_K_G = np.random.randn(d_model, d_model) * 0.1
        self.W_V_G = np.random.randn(d_model, d_model) * 0.1

        # Nota: os pesos são inicializados com valores pequenos (multiplicados por 0.1) 
        # paara evitar que o softmax colapse. Tive esse problema e perguntei a IA o que fazer, e ela me sugeriu isso.

    def forward(self, X):
        Qs, Ks, Vs = [], [], []

        # Cada cabeca i gera suas projecoes locais
        for i in range(self.h):
            Qs.append(X @ self.W_Q[i])  
            Ks.append(X @ self.W_K[i])
            Vs.append(X @ self.W_V[i])

        # Concatena todas as cabecas
        Q_cat = np.concatenate(Qs, axis=-1)
        K_cat = np.concatenate(Ks, axis=-1)
        V_cat = np.concatenate(Vs, axis=-1)

        # Mistura as perspectivas de todas as cabecas
        Q = Q_cat @ self.W_Q_G 
        K = K_cat @ self.W_K_G
        V = V_cat @ self.W_V_G

        # Z = softmax( Q @ K^T / sqrt(d_model) ) @ V
        output, _ = scaled_dot_product_attention(Q, K, V)
        return output

def layer_norm(X, epsilon=1e-6):
    # Normalização de camada: (X - mean) / sqrt(var + epsilon) (como indicado no documento)
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)


class FeedForwardNetwork:
    def __init__(self, d_model, d_ffn):
        # Inicializa as matrizes de pesos 
        self.W1 = np.random.randn(d_model, d_ffn) 
        # Inicializa os bias como zeros
        self.b1 = np.zeros(d_ffn)
        self.W2 = np.random.randn(d_ffn, d_model) 
        self.b2 = np.zeros(d_model)
 
    def forward(self, X):
        hidden = np.maximum(0, X @ self.W1 + self.b1)   # ReLU
        # A saída é a segunda transformação linear pedida
        return hidden @ self.W2 + self.b2


class EncoderBlock:
    def __init__(self, d_model, h, d_ffn):
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X):
        # O bloco do encoder descrito é composto por uma camada de multi-head attention e seguida por uma camada feed-forward,
        # com normalização depois de cada uma delas
        X_att   = self.mha.forward(X)
        X_norm1 = layer_norm(X + X_att)
        X_ffn   = self.ffn.forward(X_norm1)
        X_out   = layer_norm(X_norm1 + X_ffn)
        return X_out

