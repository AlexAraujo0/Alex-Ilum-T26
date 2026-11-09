# modelo de armazenamento: "DIRETORIO/NOME_ARQUIVO"

import pandas as pd
from otimizacao import *
from sklearn.linear_model import Ridge

df = pd.read_csv("./data_TTV/df_treino.csv")

X_treino = df.drop("critical_temp", axis=1)
y_treino = df[["critical_temp"]].values.ravel()

ensaios = estudo(nome="RIDGE_ciclo3", arquivo="E_RIDGE/RIDGE_ciclo3", carregar=True)

ensaios.optimize(lambda trial: objetivo(trial=trial, X=X_treino, y=y_treino, algoritmo=Ridge), n_trials=200)