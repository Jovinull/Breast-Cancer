# Importando bibliotecas necessárias
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as k 
from tensorflow.keras.models import Sequential

# Carregando os dados de entrada e saída do arquivo CSV
previsores = pd.read_csv('database/entradas_breast.csv')
classe = pd.read_csv('database/saidas_breast.csv')

# Função para criar a arquitetura da rede neural
def criarRede(): 
    # Limpando a sessão do TensorFlow para evitar possíveis interferências entre modelos
    k.clear_session()
    
    # Criando um modelo de rede neural sequencial
    classificador = Sequential([
        # Adicionando a primeira camada oculta com ativação ReLU
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30),
        # Adicionando uma camada de Dropout para prevenir overfitting
        tf.keras.layers.Dropout(0.2),
        # Adicionando a segunda camada oculta com ativação ReLU
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        # Adicionando uma camada de Dropout para prevenir overfitting
        tf.keras.layers.Dropout(0.2),
        # Adicionando a camada de saída com ativação sigmoid (problema de classificação binária)
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    
    # Compilando o modelo usando o otimizador Adam, função de perda binary_crossentropy e métrica binary_accuracy
    classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador

# É uma classe utilizada para envolver modelos Keras, permitindo que eles sejam utilizados com ferramentas que requerem
# uma interface de estilo scikit-learn. Essa classe provavelmente é parte da biblioteca 
classificador = KerasClassifier(model=criarRede, epochs=100, batch_size=10)

# Realizando validação cruzada (cross-validation) no modelo usando 10 folds e medindo a precisão (accuracy)
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

# Calculando a média e o desvio padrão dos resultados da validação cruzada
media = resultados.mean()
desvio = resultados.std()
