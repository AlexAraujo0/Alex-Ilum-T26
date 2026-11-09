# modelo de armazenamento: "DIRETORIO/NOME_ARQUIVO"

import pandas as pd
from otimizacao import *
from sklearn.ensemble import ExtraTreesRegressor

df = pd.read_csv("./data_TTV/df_treino.csv")

X_treino = df.drop("critical_temp", axis=1)
y_treino = df[["critical_temp"]].values.ravel()

ensaios = estudo(nome="ETR_ciclo3_corr", arquivo="E_ETR/ETR_ciclo3_corr", carregar=True)

ensaios.optimize(lambda trial: objetivo(trial=trial, X=X_treino, y=y_treino, algoritmo=ExtraTreesRegressor), n_trials=100)