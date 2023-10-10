# %%
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# %%

# Carregue seus dados em um dataframe (supondo que o nome do dataframe seja "dados")
dados = pd.read_csv("data/train_with_flight_v2.csv")
#dados = dados[dados["target"] > 300]

# %%


# %%
dados["sum(flight_through_area)"].max()

# %%
del dados["HoraData"],dados["HoraDataDest"]

# %%
del dados["flightid_"]

# %%
del dados["trocareal"],dados["troca_prev_mean_x"],dados["troca_prev_sum_x"],dados["troca_prev_mean_y"],dados["troca_prev_sum_y"],dados["troca_prev_mean"],dados["troca_prev_sum"],dados["troca_real_x"],dados["troca_real_y"],dados["troca_real"]

# %%
# Primeiro, vamos carregar os valores únicos a partir do arquivo CSV
origem = pd.read_csv('data/origem.csv')
mapping_dict = {i: val for i, val in enumerate(origem.values)}
# Agora, vamos reverter os valores codificados de volta aos valores originais
dados['origem'] = dados['origem'].map(mapping_dict)

destino = pd.read_csv('data/destino.csv')
mapping_dict = {i: val for i, val in enumerate(destino.values)}

dados['destino'] = dados['destino'].map(mapping_dict)


# %%
dados = dados[dados["origem"] != dados["destino"]]

# %%
dados["origem"] = dados["origem"].astype('str')
dados["destino"] = dados["destino"].astype('str')

# %%
dados["origem"], origem = pd.factorize(dados["origem"]) #verificar se rodando novamente muda de código
dados["destino"], destino = pd.factorize(dados["destino"])


# %%
for col in dados.columns:
    print(col)

# %%
#del dados["troca_real"], dados["troca_realtcreal_sem"], dados["troca_realb5"]


# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
X = dados.drop(["target", "flightid"], axis=1)
y = dados["target"]
sc = StandardScaler()
orderCols = X.columns
X = sc.fit_transform(X)

test = pd.read_csv("data/test_with_flight_v2.csv")
#del test["troca_prev_mean"], test["troca_prev_sum"]
del test["hora_ref"], test["dt_dep"], test["snapshot_radar"], test["path"], test["hora_esperas"], test["aero_esperas"], test["hora_metaf"], test["metaf"], test["aero_metaf"], test["hora_metar"], test["metar"], test["aero_metar"], test["hora_tcp"], test["aero_tcp"], test["hora_tcr"], test["aero_tcr"], test["eqq"], test["data"], test["first(substring(CAST(polygon AS STRING) 3 13))"], test["flight_through_area"], test["aero_x"], test["aero_y"], test["aero"], test["HoraData"], test["linha_first"], test["flightid_"]
test["sum(flight_through_area)"] = test["sum(flight_through_area)"]/446
aux = np.where(test["origem"] == test["destino"], 1, 0)
del test["flightid"]
origem = pd.read_csv('data/origem.csv')
mapping_dict = {str(val)[2:-2]: i for i, val in enumerate(origem.values)}
# Primeiro, vamos carregar os valores únicos a partir do arquivo CSV

# Agora, vamos reverter os valores codificados de volta aos valores originais
test['origem'] = test['origem'].map(mapping_dict)

destino = pd.read_csv('data/destino.csv')
mapping_dict = {str(val)[2:-2]: i for i, val in enumerate(destino.values)}

test['destino'] = test['destino'].map(mapping_dict)

destino = pd.read_csv('data/linha.csv')
mapping_dict = {str(val)[2:-2]: i for i, val in enumerate(destino.values)}

test['linha'] = test['linha'].map(mapping_dict)

#orderCols = ["origem","destino","linha","diaSemana","hora","esperas","mean_x","sum_x","mean_y","sum_y","mean","sum","Temperaturamean_x","Temperaturamin_x","Temperaturamax_x","Temperaturastd_x","Ponto de Orvalhomean_x","Ponto de Orvalhomin_x","Ponto de Orvalhomax_x","Ponto de Orvalhostd_x","Velocidade do Ventomean_x","Velocidade do Ventomin_x","Velocidade do Ventomax_x","Velocidade do Ventostd_x","Direção do Ventomean_x","Direção do Ventomin_x","Direção do Ventomax_x","Direção do Ventostd_x","Visibilidademean_x","Visibilidademin_x","Visibilidademax_x","Visibilidadestd_x","Pressãomean_x","Pressãomin_x","Pressãomax_x","Pressãostd_x","Temperaturamean_y","Temperaturamin_y","Temperaturamax_y","Temperaturastd_y","Ponto de Orvalhomean_y","Ponto de Orvalhomin_y","Ponto de Orvalhomax_y","Ponto de Orvalhostd_y","Velocidade do Ventomean_y","Velocidade do Ventomin_y","Velocidade do Ventomax_y","Velocidade do Ventostd_y","Direção do Ventomean_y","Direção do Ventomin_y","Direção do Ventomax_y","Direção do Ventostd_y","Visibilidademean_y","Visibilidademin_y","Visibilidademax_y","Visibilidadestd_y","Pressãomean_y","Pressãomin_y","Pressãomax_y","Pressãostd_y","troca","dt_radar_lambda","dt_radar_lambda1","dt_radar_lambda2","dt_radar_lambda3","flightlevel_mean","flightlevel_max","distancia_","flightlevel_std","flightlevel_std1","speed_std","speed_std1", "sum(flight_through_area)"]

#y_test = model.predict(test)
test = sc.transform(test[orderCols])
#y_test = model.predict(test, num_iteration=model.best_iteration)



# %%
nfolds = 10
kf = KFold(n_splits=nfolds, shuffle=True, random_state=42) # you can change 'n_splits' to define how many folds you want

auxtest = pd.DataFrame()

auxtest["ID"] = pd.read_csv("data/test_with_flight_v2.csv")["flightid"]
auxtest["solution"] = 0
mse_list = []

for train_indices, test_indices in kf.split(X):
    X_train, y_train = X[train_indices], y.iloc[train_indices]
    X_test, y_test =   X[test_indices],  y.iloc[test_indices]

    d_train = lgb.Dataset(X_train, label=y_train)

    params = {'objective': 'regression',
              'metric': "rmse",  # try rmse
              'num_leaves': 31,
              'learning_rate': 0.05,
              'feature_fraction': 0.9,
              'bagging_fraction': 0.8,
              'bagging_freq': 5}

    model = lgb.train(params, d_train, num_boost_round=1000, valid_sets=[d_train],
                      callbacks=([lgb.early_stopping(stopping_rounds=10)]))

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse_list.append(rmse)
    print(f"RMSE: {rmse}")
    y_test = model.predict(test, num_iteration=model.best_iteration)
    auxtest["solution"] = auxtest["solution"] + y_test
    
auxtest["solution"] = auxtest["solution"] / nfolds
auxtest["solution"] = np.where(aux == 1, 0, auxtest["solution"])
auxtest[['ID', 'solution']].to_csv('data/submission_flight_through_area_10folds.csv', index=False)
print("Mean RMSE: ", np.mean(mse_list))
