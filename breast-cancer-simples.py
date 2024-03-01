# Importando bibliotecas necess�rias
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

# Carregando os dados de entrada e sa�da do arquivo CSV
previsores = pd.read_csv('database/entradas_breast.csv')
classe = pd.read_csv('database/saidas_breast.csv')

# Dividindo os dados em conjuntos de treinamento e teste (75% treinamento, 25% teste)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

# Criando um modelo de rede neural sequencial
classificador = Sequential()

# Adicionando a primeira camada oculta com ativa��o ReLU
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))

# Adicionando a segunda camada oculta com ativa��o ReLU
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))

# Adicionando a camada de sa�da com ativa��o sigmoid (problema de classifica��o bin�ria)
classificador.add(Dense(units=1, activation='sigmoid'))

# Compilando o modelo usando o otimizador Adam, fun��o de perda binary_crossentropy e m�trica binary_accuracy
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Treinando o modelo com os dados de treinamento, utilizando batch_size=10 e 100 epochs
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

# Obtendo os pesos das camadas do modelo
pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

# Fazendo previs�es com dados de teste
previsoes = classificador.predict(previsores_teste)

# Transformando as previs�es em valores bin�rios (True ou False) com um limiar de 0.5
previsoes = (previsoes > 0.5)

# Calculando a precis�o do modelo comparando as previs�es com os valores reais
precisao = accuracy_score(classe_teste, previsoes)

# Criando a matriz de confus�o para avaliar o desempenho do modelo
matriz = confusion_matrix(classe_teste, previsoes)

# Avaliando o desempenho do modelo nos dados de teste utilizando a fun��o evaluate.
# A fun��o evaluate retorna uma lista contendo a perda (loss) e as m�tricas definidas durante a compila��o do modelo,
# neste caso, a precis�o (accuracy). Os valores retornados s�o a m�dia dessas m�tricas calculadas sobre o conjunto de teste.
resultado = classificador.evaluate(previsores_teste, classe_teste)
# O resultado[0] cont�m a perda (loss) e o resultado[1] cont�m a precis�o (accuracy).
