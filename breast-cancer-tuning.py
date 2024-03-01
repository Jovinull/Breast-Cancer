import pandas as pd
from tensorflow.keras.models import Sequential
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k

# Carregando as bibliotecas necessárias
# Nota: Certifique-se de ter as versões corretas das bibliotecas TensorFlow e Scikeras instaladas.

# Carregando os dados de entrada e saída do arquivo CSV
previsores = pd.read_csv('database/entradas_breast.csv')
classe = pd.read_csv('database/saidas_breast.csv')

# Função para criar a arquitetura da rede neural
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    # Limpando a sessão do TensorFlow para evitar possíveis interferências entre modelos
    k.clear_session()
    
    # Criando um modelo de rede neural sequencial
    classificador = Sequential([
               # Adicionando a primeira camada oculta com ativação ReLU
               tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30),
               # Adicionando uma camada de Dropout para prevenir overfitting
               tf.keras.layers.Dropout(0.2),
               # Adicionando a segunda camada oculta com ativação ReLU
               tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
               # Adicionando uma camada de Dropout para prevenir overfitting
               tf.keras.layers.Dropout(0.2),
               # Adicionando a camada de saída com ativação sigmoid (problema de classificação binária)
               tf.keras.layers.Dense(units=1, activation='sigmoid')])
    
    # Compilando o modelo usando o otimizador, função de perda e métrica especificados
    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return classificador

# Criando uma instância do KerasClassifier, envolvendo o modelo criado pela função
classificador = KerasClassifier(build_fn=criarRede)

# Definindo os parâmetros para a busca em grade (Grid Search)
parametros = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [16, 8]
}

# Criando uma instância do GridSearchCV para encontrar os melhores parâmetros
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           scoring='accuracy',
                           cv=5)

# Realizando a busca em grade no conjunto de dados
grid_search = grid_search.fit(previsores, classe)

# Obtendo os melhores parâmetros e a melhor precisão
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
