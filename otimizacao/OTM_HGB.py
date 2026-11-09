# modelo de armazenamento: "DIRETORIO/NOME_ARQUIVO"

import pandas as pd
from otimizacao import *
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_csv("./data_TTV/df_treino.csv")

X_treino = df.drop("critical_temp", axis=1)
y_treino = df[["critical_temp"]].values.ravel()

ensaios = estudo(nome="HGB_ciclo3", arquivo="E_HGB/HGB_ciclo3", carregar=True)

ensaios.optimize(lambda trial: objetivo(trial=trial, X=X_treino, y=y_treino, algoritmo=HistGradientBoostingRegressor), n_trials=100)