import pandas as pd
from otimizacao import *

df = pd.read_csv("./data_TTV/df_treino.csv")

X_treino = df.drop("critical_temp", axis=1)
y_treino = df["critical_temp"].values

ensaios = estudo(nome="Exodia_otm", arquivo="E_EXODIA/Exodia_otm", carregar=True)

ensaios.optimize(
    lambda trial: objetivo_stack(trial, X_treino, y_treino),
    n_trials=100,
)