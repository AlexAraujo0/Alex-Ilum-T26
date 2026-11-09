# Básicos
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import os

# Algoritmos de ML
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor


# Tratamento de dados
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler

# Métricas
from sklearn.metrics import root_mean_squared_error, make_scorer

SEMENTE = 12151822

# Instanciando algoritmos

def instanciar_alg(algoritmo, trial: optuna.Trial):
    if algoritmo is HistGradientBoostingRegressor:
        parametros = {
            'learning_rate' : trial.suggest_float('learning_rate', 0.05, 0.08, log=True),
            'max_depth' : trial.suggest_int('max_depth', 9, 15),
            'random_state' : SEMENTE,
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 12, 20),
            'max_iter' : trial.suggest_int('max_iter', 700, 830),
            'l2_regularization' : trial.suggest_float('l2_regularization', 1e-7, 1.2, log=True),
            }
        return HistGradientBoostingRegressor(**parametros)

    elif algoritmo is KNeighborsRegressor:
        parametros = {
            'n_neighbors' : trial.suggest_int('n_neighbors', 2, 10),
            'weights' : 'distance',
            'p' : 1,
            'n_jobs' : 2
        }
        pp = Pipeline([("Normalizador", instanciar_normalizador(trial)),
                      ("Modelo", KNeighborsRegressor(**parametros))])
        return pp

    elif algoritmo is Ridge:
        parametros = {
            'alpha' : trial.suggest_float('alpha', 0.01, 0.09, log=True),
            'fit_intercept' : True,
            'solver' : trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky']),
            'random_state' : SEMENTE,
        }

        pp = Pipeline([("Normalizador", instanciar_normalizador(trial)),
                       ("Modelo", Ridge(**parametros))])
        return pp

    elif algoritmo is ExtraTreesRegressor:
        parametros = {
            'n_estimators' : trial.suggest_int('n_estimators', 400, 1000, log=True),
            'max_depth' : trial.suggest_int('max_depth', 10, 45),
            'random_state' : SEMENTE,
            'n_jobs' : 6,
            'max_features' : trial.suggest_float('max_features', 0.7, 1.0, log=True),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5),
            'min_samples_split' : trial.suggest_int('min_samples_split', 3, 10),
        }
        return ExtraTreesRegressor(**parametros)

    else: return DummyRegressor()

# Normalizador
def instanciar_normalizador(trial: optuna.Trial) -> RobustScaler:
  parametros = {
    'with_centering': trial.suggest_categorical('with_centering', [True, False]),
    'with_scaling': trial.suggest_categorical('with_scaling', [True, False])
  }
  return RobustScaler(**parametros)

# Redução de dimensionalidade
def instanciar_Percentil(trial: optuna.Trial) -> SelectPercentile:
    parametros = {
        "percentile": trial.suggest_int("percentile", 70, 100, step=5),
        "score_func": f_regression
    }
    return SelectPercentile(**parametros)

# Pipeline do modelo
def instanciar_modelo(algoritmo, trial: optuna.Trial) -> Pipeline:
    seletor = instanciar_Percentil(trial)

    alg_instanciado = instanciar_alg(algoritmo, trial)

    modelo = Pipeline([
        ("Seletor", seletor),
        ("Modelo", alg_instanciado)
        ])

    return modelo

# Função objetivo
def objetivo(trial : optuna.Trial, X, y, algoritmo, semente=SEMENTE) -> float:
  modelo = instanciar_modelo(algoritmo=algoritmo, trial=trial)
  
  kfold = KFold(n_splits=5, shuffle=True, random_state=semente)
  estimador_RMSE = make_scorer(score_func=root_mean_squared_error, greater_is_better=False)
  pontuacao = cross_val_score(modelo, X, y, scoring=estimador_RMSE, cv=kfold, error_score='raise')
  
  return np.min([np.mean(pontuacao), np.median(pontuacao)])

# Estudo
def estudo(nome, arquivo, carregar=False):
    BASE = "/home/alex2610082/work/Tiamat"

    return optuna.create_study(
        study_name=nome,
        direction="maximize",
        storage=f"sqlite:///{BASE}/{arquivo}.db",
        load_if_exists=carregar
    )

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 3))

def exodia(trial: optuna.Trial, cv=3) -> StackingRegressor:
    knn = Pipeline([
        ('normalizador', RobustScaler()),
        ('modelo', KNeighborsRegressor(
            n_neighbors=trial.suggest_int('n_neighbors_KNN', 3, 12),
            p=trial.suggest_categorical('p_KNN', [1, 2]),
            weights='distance')),
    ])

    hgb = HistGradientBoostingRegressor(
        learning_rate=trial.suggest_float('learning_rate_HGB', 0.05, 0.09, log=True),
        max_depth=trial.suggest_int('max_depth_HGB', 7, 12),
        min_samples_leaf=trial.suggest_int('min_samples_leaf_HGB', 8, 20),
        l2_regularization=trial.suggest_float('l2_regularization_HGB', 1e-5, 1.0, log=True),
        max_iter=trial.suggest_int('max_iter_HGB', 600, 900),
        early_stopping=False,
        random_state=SEMENTE)

    etr = ExtraTreesRegressor(
        n_estimators=300,
        max_features=trial.suggest_float('max_features_ETR', 0.55, 0.9),
        min_samples_split=trial.suggest_int('min_samples_split_ETR', 4, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf_ETR', 1, 2),
        max_depth=trial.suggest_int('max_depth_ETR', 30, 50),
        n_jobs=N_JOBS, random_state=SEMENTE)

    ridge = Ridge(
        alpha=trial.suggest_float('alpha_RD', 1e-3, 1.0, log=True),
        positive=trial.suggest_categorical('positive_RD', [False, True]),
        random_state=SEMENTE)

    return StackingRegressor(
        estimators=[('KNN', knn), ('HGB', hgb), ('ETR', etr)],
        final_estimator=ridge,
        passthrough=trial.suggest_categorical('passthrough_FINAL', [True, False]),
        n_jobs=1, cv=cv)

def objetivo_stack(trial, X, y, semente=SEMENTE):
    modelo = exodia(trial, cv=3)
    kfold = KFold(n_splits=5, shuffle=True, random_state=semente)

    rmse = make_scorer(root_mean_squared_error, greater_is_better=False)
    pontos = cross_val_score(modelo, X, y, scoring=rmse, cv=kfold, error_score='raise')

    return np.min([np.mean(pontos), np.median(pontos)])