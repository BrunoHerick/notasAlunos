#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as grafico
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# OBJECTIVE: predict the likelihood of a new order being rejected
dados = pd.read_csv("/home/brunohelghast/PROFISSIONAL/PYTHON/SCIKIT_LEARN/notas_alunos/dados_notas_alunos.csv")

notas_alunos = dados.iloc[:,0:5].values
situacao_final = dados.iloc[:,5].values
aux1 = dados.iloc[:,0:5].values

label_encoder_portugues = LabelEncoder()
label_encoder_ciencia = LabelEncoder()
label_encoder_matematica = LabelEncoder()
label_encoder_historia = LabelEncoder()
label_encoder_geografia = LabelEncoder()

notas_alunos[:,0] = label_encoder_portugues.fit_transform(notas_alunos[:,0])
notas_alunos[:,1] = label_encoder_ciencia.fit_transform(notas_alunos[:,1])
notas_alunos[:,2] = label_encoder_matematica.fit_transform(notas_alunos[:,2])
notas_alunos[:,3] = label_encoder_historia.fit_transform(notas_alunos[:,3])
notas_alunos[:,4] = label_encoder_geografia.fit_transform(notas_alunos[:,4])

""" for i in range(0, len(notas_alunos[:,0]),1):
    print(notas_alunos[i,:],aux1[i,:]) """

naive_notas_alunos = GaussianNB()
naive_notas_alunos.fit(notas_alunos, situacao_final)# treinar

# combinacoes = [146,73,26,53,60,3,18,13,1]
preverSituacao = naive_notas_alunos.predict([[1,0,1,2,2]])
print(preverSituacao)