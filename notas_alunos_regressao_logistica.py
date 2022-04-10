# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

arquivo = pd.read_csv("/home/bhs/PROFISSIONAL/PYTHON/SCIKIT_LEARN/notas_alunos/dados_notas_alunos.csv")
#print(arquivo.sample(5))

arquivo = pd.get_dummies(arquivo, drop_first=False)# get_dummies() --> transforma as variaveis categoricas em binarios 1 ou 0.
#print(arquivo)

entrada = arquivo.drop("SITUACAO_aprovado",axis=1)
entrada = entrada.drop("SITUACAO_reprovado",axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(entrada, arquivo["SITUACAO_reprovado"])

regressao_logistica = LogisticRegression(solver = "lbfgs")
regressao_logistica.fit(x_treino, y_treino)# treinar modelo

paraPrever = np.array([[1,0,0,1,0,0,0,0,1,0,1,0,0,0,1]])# para cada materia, sera criada uma coluna para cada nota. Ex: ciencia_ns, ciencia_s, ciencia_p. É necessário preencher os valores de entrada para cada coluna.

previsao = regressao_logistica.predict(paraPrever)

print(previsao)# retorna o resultado para SITUACAO_reprovado (1 para reprovado, 0 para aprovado)