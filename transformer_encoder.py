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
d_ff    = 256 # Dimensão da camada feed-forward
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