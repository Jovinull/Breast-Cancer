# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

# Carregando os dados de entrada e saída do arquivo CSV
previsores = pd.read_csv('database/entradas_breast.csv')
classe = pd.read_csv('database/saidas_breast.csv')

# Dividindo os dados em conjuntos de treinamento e teste (75% treinamento, 25% teste)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

# Criando um modelo de rede neural sequencial
classificador = Sequential()

# Adicionando a primeira camada oculta com ativação ReLU
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))

# Adicionando a segunda camada oculta com ativação ReLU
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))

# Adicionando a camada de saída com ativação sigmoid (problema de classificação binária)
classificador.add(Dense(units=1, activation='sigmoid'))

# Compilando o modelo usando o otimizador Adam, função de perda binary_crossentropy e métrica binary_accuracy
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Treinando o modelo com os dados de treinamento, utilizando batch_size=10 e 100 epochs
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

# Obtendo os pesos das camadas do modelo
pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

# Fazendo previsões com dados de teste
previsoes = classificador.predict(previsores_teste)

# Transformando as previsões em valores binários (True ou False) com um limiar de 0.5
previsoes = (previsoes > 0.5)

# Calculando a precisão do modelo comparando as previsões com os valores reais
precisao = accuracy_score(classe_teste, previsoes)

# Criando a matriz de confusão para avaliar o desempenho do modelo
matriz = confusion_matrix(classe_teste, previsoes)

# Avaliando o desempenho do modelo nos dados de teste utilizando a função evaluate.
# A função evaluate retorna uma lista contendo a perda (loss) e as métricas definidas durante a compilação do modelo,
# neste caso, a precisão (accuracy). Os valores retornados são a média dessas métricas calculadas sobre o conjunto de teste.
resultado = classificador.evaluate(previsores_teste, classe_teste)
# O resultado[0] contém a perda (loss) e o resultado[1] contém a precisão (accuracy).
