# Breast Cancer Classificação Binária com Redes Neurais

## Sobre o Repositório

Este repositório contém códigos para resolver o problema de classificação de câncer de mama usando redes neurais. O conjunto de dados utilizado foi obtido do repositório de aprendizado de máquina da UCI. A solução é implementada em três scripts, cada um demonstrando uma abordagem diferente para resolver o problema.

### 1. breast-cancer-simples

Este script implementa uma rede neural simples usando a biblioteca Keras. Ele carrega o conjunto de dados, divide-o em conjuntos de treinamento e teste, constrói um modelo sequencial de rede neural com duas camadas ocultas e uma camada de saída, compila o modelo e o treina nos dados de treinamento. O script também avalia o modelo nos dados de teste, calcula a precisão e cria uma matriz de confusão para a avaliação de desempenho.

Funções de perda: binary_crossentropy
Função de ativação: relu (camadas ocultas), sigmoid (camada de saída)

### 2. breast-cancer-cruzada

Este script demonstra a validação cruzada para o modelo de rede neural. Utiliza a biblioteca scikeras para envolver o modelo Keras, permitindo seu uso com as ferramentas de validação cruzada do scikit-learn. O script define uma função para criar a arquitetura da rede neural, configura o KerasClassifier e realiza a validação cruzada com 10 folds, medindo a precisão.

Funções de perda: binary_crossentropy
Função de ativação: relu (camadas ocultas), sigmoid (camada de saída)

### 3. breast-cancer-tuning

O terceiro script foca na sintonização de hiperparâmetros usando o GridSearchCV. Explora diferentes combinações de hiperparâmetros, como tamanho do lote, número de épocas, otimizador, função de perda, inicializador de kernel, função de ativação e número de neurônios. A melhor combinação de hiperparâmetros e a precisão correspondente são então determinadas.

Funções de perda: binary_crossentropy
Função de ativação: relu (camadas ocultas), sigmoid (camada de saída)

## Ambiente de Desenvolvimento

O código foi desenvolvido no ambiente Anaconda com o Spyder como IDE.

## Conjunto de Dados UCI - Breast Cancer

O conjunto de dados utilizado é proveniente do Instituto de Oncologia e apareceu repetidamente na literatura de aprendizado de máquina. Ele consiste em 286 instâncias, com 201 pertencendo a uma classe e 85 a outra. Os atributos incluem dados lineares e nominais relacionados a características de câncer de mama.

Referência: [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/14/breast+cancer)

## Curso de Deep Learning com Python de A a Z

Este projeto foi desenvolvido como parte do curso "Deep Learning com Python de A a Z - O Curso Completo" da IA Expert Academy. O curso fornece uma compreensão abrangente de deep learning e python, e você pode encontrá-lo [aqui](https://www.udemy.com/course/deep-learning-com-python-az-curso-completo/?couponCode=KEEPLEARNING).
