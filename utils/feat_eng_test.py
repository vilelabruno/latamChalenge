# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

test = pd.read_csv("data/idsc_dataset.csv", delimiter=";")

# %%
test["eqq"] = np.where(test["aero_metar"] == test["destino"], 1, 0)

# %%
test[['aero_metar', 'destino', 'eqq']]

# %%
test["data"] = pd.to_datetime(test["dt_dep"])

# %%
flight_throught = pd.read_csv("data/test_flight_througt.csv")

# %%
test = test.merge(flight_throught, on="flightid", how="left")

# %%
def checkCols(df):
    checkArr = ["flightid","origem","destino","target","linha","diaSemana","hora","esperas","mean_x","sum_x","mean_y","sum_y","mean","sum","Temperaturamean_x","Temperaturamin_x","Temperaturamax_x","Temperaturastd_x","Ponto de Orvalhomean_x","Ponto de Orvalhomin_x","Ponto de Orvalhomax_x","Ponto de Orvalhostd_x","Velocidade do Ventomean_x","Velocidade do Ventomin_x","Velocidade do Ventomax_x","Velocidade do Ventostd_x","Direção do Ventomean_x","Direção do Ventomin_x","Direção do Ventomax_x","Direção do Ventostd_x","Visibilidademean_x","Visibilidademin_x","Visibilidademax_x","Visibilidadestd_x","Pressãomean_x","Pressãomin_x","Pressãomax_x","Pressãostd_x","Temperaturamean_y","Temperaturamin_y","Temperaturamax_y","Temperaturastd_y","Ponto de Orvalhomean_y","Ponto de Orvalhomin_y","Ponto de Orvalhomax_y","Ponto de Orvalhostd_y","Velocidade do Ventomean_y","Velocidade do Ventomin_y","Velocidade do Ventomax_y","Velocidade do Ventostd_y","Direção do Ventomean_y","Direção do Ventomin_y","Direção do Ventomax_y","Direção do Ventostd_y","Visibilidademean_y","Visibilidademin_y","Visibilidademax_y","Visibilidadestd_y","Pressãomean_y","Pressãomin_y","Pressãomax_y","Pressãostd_y","troca","trocareal","troca_prev_meanb2","troca_prev_sumb2","troca_prev_meantcprev_sem","troca_prev_sumtcprev_sem","troca_realb5","troca_realtcreal_sem","troca_real","dt_radar_lambda","dt_radar_lambda1","dt_radar_lambda2","dt_radar_lambda3","flightlevel_mean","flightlevel_max","distancia_","flightlevel_std","flightlevel_std1","speed_std","speed_std1","sum(flight_through_area)"]

    for col in checkArr:
        if col not in df.columns:
            print(col)

# %%
test["data"] = pd.to_datetime(test["path"].str.split("/").str[-1].str.split("_", expand=True)[1].str.split(".", expand=True)[0])

# %%
test["diaSemana"] = test["data"].dt.dayofweek

# %%
test["linha"] = test["origem"] + test["destino"]
test["hora"] = test["data"].dt.hour


# %%
esperas_hora = pd.read_csv("data/esperas_hora.csv")
esperas_semana = pd.read_csv("data/esperas_semana.csv")
esperas_aero  = pd.read_csv("data/esperas_aero.csv")

# %%
print(len(test))
test = test.merge(esperas_hora, right_on=["aero", "hora"], left_on=["origem", "hora"] , how="left")
test = test.merge(esperas_semana, right_on=["aero", "diaSemana"], left_on=["origem", "diaSemana"] , how="left")
test = test.merge(esperas_aero, right_on=["aero"], left_on=["origem"] , how="left")
print(len(test))

# %%
metar = pd.read_csv("data/test_metar.csv")

# %%
import pandas as pd
from metar.Metar import Metar

# Função para parsear METAR
def parse_metar(metar):
    metar = metar.replace('METAF', 'METAR') 
    try:
        metar_obj = Metar(metar)
        temperature = metar_obj.temp.value() if metar_obj.temp else None
        dewpoint = metar_obj.dewpt.value() if metar_obj.dewpt else None
        wind_speed = metar_obj.wind_speed.value() if metar_obj.wind_speed else None
        wind_direction = metar_obj.wind_dir.value() if metar_obj.wind_dir else None
        visibility = metar_obj.vis.value() if metar_obj.vis else None
        pressure = metar_obj.press.value() if metar_obj.press else None
        weather = ', '.join([str(c) for c in metar_obj.weather]) if metar_obj.weather else None
        sky = ', '.join([str(c) for c in metar_obj.sky]) if metar_obj.sky else None
        return pd.Series([temperature, dewpoint, wind_speed, wind_direction, visibility, pressure, weather, sky])
    except Exception as e:
        #print(str(e))
        return pd.Series([None, None, None, None, None, None, None, None])

# Aplicar parse sobre METAR para cada linha
metar[['Temperatura', 'Ponto de Orvalho', 'Velocidade do Vento', 'Direção do Vento', 'Visibilidade', 'Pressão', 'Tempo', 'Céu']] = metar['metar'].apply(parse_metar)


# %%
metar.fillna(-99, inplace=True)

# %%

metar['HoraData'] = pd.to_datetime(metar['data']).dt.strftime('%Y-%m-%d %H')
# Mesclar os dataframes usando a coluna "HoraData"

# %%
test['HoraData'] = pd.to_datetime(test['data']).dt.strftime('%Y-%m-%d %H')

# %%


# %%
# Agrupando por flight_id
df_merge = pd.merge(test, metar, left_on=['destino', 'HoraData'], right_on=['aero_metar', 'HoraData'], how='left')

grouped_df = df_merge.groupby('flightid')

# Calculando as estatísticas
stats_df = grouped_df.agg({
    'Temperatura': ['mean', 'min', 'max', 'std'],
    'Ponto de Orvalho': ['mean', 'min', 'max', 'std'],
    'Velocidade do Vento': ['mean', 'min', 'max', 'std'],
    'Direção do Vento': ['mean', 'min', 'max', 'std'],
    'Visibilidade': ['mean', 'min', 'max', 'std'],
    'Pressão': ['mean', 'min', 'max', 'std']
})
stats_df.reset_index(inplace=True)
stats_df.columns = [''.join(col) for col in stats_df.columns]
test = test.merge(stats_df, on="flightid", how="left")

# %%

metaf = pd.read_csv("data/test_metaf.csv")
metaf['metaf'] = metaf['metaf'].apply(lambda x: ' '.join(x.split()))

metaf[['Temperatura', 'Ponto de Orvalho', 'Velocidade do Vento', 'Direção do Vento', 'Visibilidade', 'Pressão', 'Tempo', 'Céu']] = metaf['metaf'].apply(parse_metar)

test

# %%

metaf['HoraData'] = pd.to_datetime(metaf['data']).dt.strftime('%Y-%m-%d %H')
# Mesclar os dataframes usando a coluna "HoraData"


# %%
metaf

# %%

df_merge = pd.merge(test, metaf, left_on=['destino', 'HoraData'], right_on=['aero_metaf', 'HoraData'], how='left')
# Agrupando por flight_id
grouped_df = df_merge.groupby('flightid')

# Calculando as estatísticas
stats_df = grouped_df.agg({
    'Temperatura': ['mean', 'min', 'max', 'std'],
    'Ponto de Orvalho': ['mean', 'min', 'max', 'std'],
    'Velocidade do Vento': ['mean', 'min', 'max', 'std'],
    'Direção do Vento': ['mean', 'min', 'max', 'std'],
    'Visibilidade': ['mean', 'min', 'max', 'std'],
    'Pressão': ['mean', 'min', 'max', 'std']
})
stats_df.columns = [''.join(col) for col in stats_df.columns]
stats_df.reset_index(inplace=True)

test = test.merge(stats_df, on="flightid", how="left")
# Preencher NaN na coluna de espera com 0 (sem espera)
test

# %%


# %%
for col in test.columns:
    print(col)

# %%
test[["hora_tcp" ,"troca" ,"aero_tcp" ,"hora_tcr" ,"aero_tcr"]]

# %%
resumo = pd.read_csv("data/resumo_voo.csv")

# %%
resumo

# %%

test = test.merge(resumo, right_on="linha_first", left_on="linha", how="left")

# %%
test

# %%


# %%
test_flight_through = pd.read_csv("data/test_flight_througt.csv")

# %%
test_flight_through["sum(flight_through_area)"] = test_flight_through["flight_through_area"]

# %%
test = test.merge(test_flight_through[["flightid", "sum(flight_through_area)"]], on="flightid", how="left")

# %%
#test.fillna(-99, inplace=True)

# %%
checkCols(test)

# %%
import re
test.columns = [re.sub('[\[\]<>\'"\'.,]', '', col) for col in test.columns]

# %%
test.to_csv("data/test_with_flight_v2.csv", index=False)


