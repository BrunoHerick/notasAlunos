#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as grafico
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#OBJECTIVE: predict the likelihood of a new order being rejected

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

arvore_notas_alunos = DecisionTreeClassifier(criterion="entropy")
arvore_notas_alunos.fit(notas_alunos, situacao_final)

#print(arvore_situacao_alunos.feature_importances_) --> [0.227 0.133 0.37 0.267 0. ] = ganhos de informação, matematica tem o maior ganho de informacao com 0.37
""" MOSTRAR ARVORE 
previsores = ["PORTUGUES","CIENCIA","MATEMATICA","HISTORIA","GEOGRAFIA","SITUACAO"]
figura, eixos = grafico.subplots(nrows=1, ncols=1,figsize=[10,10])
tree.plot_tree(arvore_situacao_alunos, feature_names=previsores, class_names=arvore_situacao_alunos.classes, filled=True) """

previsoes = arvore_notas_alunos.predict([[1,0,1,2,2]])
print(previsoes)