# Importando bibliotecas necess�rias
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as k 
from tensorflow.keras.models import Sequential

# Carregando os dados de entrada e sa�da do arquivo CSV
previsores = pd.read_csv('database/entradas_breast.csv')
classe = pd.read_csv('database/saidas_breast.csv')

# Fun��o para criar a arquitetura da rede neural
def criarRede(): 
    # Limpando a sess�o do TensorFlow para evitar poss�veis interfer�ncias entre modelos
    k.clear_session()
    
    # Criando um modelo de rede neural sequencial
    classificador = Sequential([
        # Adicionando a primeira camada oculta com ativa��o ReLU
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30),
        # Adicionando uma camada de Dropout para prevenir overfitting
        tf.keras.layers.Dropout(0.2),
        # Adicionando a segunda camada oculta com ativa��o ReLU
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        # Adicionando uma camada de Dropout para prevenir overfitting
        tf.keras.layers.Dropout(0.2),
        # Adicionando a camada de sa�da com ativa��o sigmoid (problema de classifica��o bin�ria)
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    
    # Compilando o modelo usando o otimizador Adam, fun��o de perda binary_crossentropy e m�trica binary_accuracy
    classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador

# � uma classe utilizada para envolver modelos Keras, permitindo que eles sejam utilizados com ferramentas que requerem
# uma interface de estilo scikit-learn. Essa classe provavelmente � parte da biblioteca 
classificador = KerasClassifier(model=criarRede, epochs=100, batch_size=10)

# Realizando valida��o cruzada (cross-validation) no modelo usando 10 folds e medindo a precis�o (accuracy)
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

# Calculando a m�dia e o desvio padr�o dos resultados da valida��o cruzada
media = resultados.mean()
desvio = resultados.std()
