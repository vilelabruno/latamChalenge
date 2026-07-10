# %%
import pandas as pd
import re

# %%
cat62 = pd.read_csv("data/cat-62.csv")

# %%

#cat62 = cat62[cat62["flightid"] != "Data final não disponível"]

# %%


cat62['dt_radar'] = pd.to_datetime(cat62['dt_radar'], unit='ms')

# %%
bimtra = pd.read_csv("data/bimtra.csv")
bimtra['dt_dep'] = pd.to_datetime(bimtra['dt_dep'], unit='ms')
bimtra['dt_arr'] = pd.to_datetime(bimtra['dt_arr'], unit='ms')

# %%
bimtra["target"] = (bimtra["dt_arr"] - bimtra["dt_dep"]) // pd.Timedelta('1s')

# %%
bimtra["linha"] = bimtra["origem"] + bimtra["destino"]

# %%
bimtra = bimtra.drop_duplicates(subset='flightid')

# %%
#bimtra = bimtra[bimtra["target"] > 1500]

# %%
bimtra

# %%
cat62 = cat62.merge(bimtra, on="flightid", how="left")

# %%
cat62["linha"] = cat62["origem"] + cat62["destino"]

# %%
cat62.sort_values(by=['flightid', 'dt_radar'], inplace=True)

# %%
from math import radians, sin, cos, sqrt, atan2, degrees

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # raio da Terra em km

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# %%


cat62["lat"] = cat62["lat"].apply(degrees)
cat62["lon"] = cat62["lon"].apply(degrees)


# %%
resumo_voo = cat62.groupby('flightid').agg({
    'dt_radar': lambda x: (x.max() - x.min()).total_seconds(),
    'lat': ['first', 'last'],
    'lon': ['first', 'last'],
    'flightlevel': ['mean', 'min', 'max', 'std'],
    'speed': ['mean', 'min', 'max', 'std'],
    'linha': 'first'
})


# %%
resumo_voo

# %%
resumo_voo['distancia'] = resumo_voo.apply(lambda row: haversine(row[('lat', 'first')], row[('lon', 'first')], 
                                                           row[('lat', 'last')], row[('lon', 'last')]), axis=1)


# %%
# Aplicar a função join às colunas do DataFrame
resumo_voo.reset_index(inplace=True)


# %%
resumo_voo

# %%
resumo_voo = resumo_voo.T.reset_index().T

# %%


# %%
arr = []
for i in range(0, len(resumo_voo.iloc[0].values)):
    arr.append(resumo_voo.iloc[0].values[i] + "_" + resumo_voo.iloc[1].values[i])

# %%
resumo_voo = resumo_voo.rename(columns=dict(zip(resumo_voo.columns, arr)))


# %%
resumo_voo = resumo_voo.iloc[2:]

# %%
resumo_voo.columns

# %%
resumo = resumo_voo.groupby('linha_first').agg({
    'flightid_': 'count',
    'dt_radar_<lambda>': ['mean', 'min', 'max', 'std'],
    'flightlevel_mean': 'mean',
    'flightlevel_max': 'mean',
    'distancia_': 'mean',
    'flightlevel_std': ['std', 'mean'],
    'speed_std': ['mean','std']
})

# Exibir o resumo
print(resumo)

# %%
resumo.reset_index(inplace=True)
resumo.to_csv("data/resumo_voo.csv", index=False)

# %%
import pandas as pd
resumo = pd.read_csv("data/resumo_voo.csv")
resumo

# %%
resumo

# %%
import pandas as pd
esperas = pd.read_csv("data/esperas.csv")
esperas["dt_dep"] = pd.to_datetime(esperas["hora"], unit='ms')
esperas["hora"] = pd.to_datetime(esperas["hora"], unit='ms')
esperas["diaSemana"] = esperas["hora"].dt.dayofweek

esperas["hora"] = esperas["hora"].dt.hour

# %%
esperas_hora = esperas.groupby(["aero", "hora"]).agg({"esperas": ["mean", "sum"]})
esperas_semana = esperas.groupby(["aero", "diaSemana"]).agg({"esperas": ["mean", "sum"]})

# %%

esperas_aero = esperas.groupby(["aero"]).agg({"esperas": ["mean", "sum"]})

# %%
esperas

# %%
bimtra["diaSemana"] = bimtra["dt_dep"].dt.dayofweek

bimtra["hora"] = bimtra["dt_dep"].dt.hour

# %%
bimtra['HoraDataDest'] = bimtra['dt_arr'].dt.strftime('%Y-%m-%d %H')

# %%
#esperas.to_csv("data/esperas_join.csv", index=False)

# %%
bimtra.shape

# %%
bimtra['HoraData'] = bimtra['dt_dep'].dt.strftime('%Y-%m-%d %H')
esperas['HoraData'] = esperas['dt_dep'].dt.strftime('%Y-%m-%d %H')

# Mesclar os dataframes usando a coluna "HoraData"
df_merge = pd.merge(bimtra, esperas, on='HoraData', how='left')

# Preencher NaN na coluna de espera com 0 (sem espera)
df_merge['esperas'] = df_merge['esperas'].fillna(0)
df_merge = df_merge.groupby("flightid").agg({"esperas": "sum"})
bimtra = bimtra.merge(df_merge.reset_index(), on="flightid", how="left")

# %%
bimtra.shape

# %%
esperas_hora.columns = esperas_hora.columns.droplevel(0)

# %%
esperas_hora.reset_index(inplace=True)

# %%
esperas_semana.columns = esperas_semana.columns.droplevel(0)
esperas_semana.reset_index(inplace=True)

# %%
esperas_aero.columns = esperas_aero.columns.droplevel(0)
esperas_aero.reset_index(inplace=True)

# %%
esperas_aero

# %%
esperas_semana

# %%
esperas_hora.to_csv("data/esperas_hora.csv", index=False)
esperas_semana.to_csv("data/esperas_semana.csv", index=False)

# %%
bimtra.shape

# %%
bimtra = bimtra.merge(esperas_hora, right_on=["aero", "hora"], left_on=["origem", "hora"] , how="left")
bimtra = bimtra.merge(esperas_semana, right_on=["aero", "diaSemana"], left_on=["origem", "diaSemana"] , how="left")

# %%
bimtra.shape

# %%
esperas_aero.to_csv("data/esperas_aero.csv", index=False)

# %%
bimtra.shape

# %%
bimtra = bimtra.merge(esperas_aero, right_on=["aero"], left_on=["origem"] , how="left")

# %%
bimtra.shape

# %%
#import pandas as pd
metar = pd.read_csv("data/metar.csv")

# %%
metar

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
metar

# %%
metar.iloc[0]["metar"]

# %%

metar['HoraData'] = pd.to_datetime(metar['hora'], unit='ms').dt.strftime('%Y-%m-%d %H')
# Mesclar os dataframes usando a coluna "HoraData"



# Preencher NaN na coluna de espera com 0 (sem espera)


# %%
del bimtra["aero"]

# %%
# Agrupando por flight_id
df_merge = pd.merge(bimtra, metar, left_on=['destino', 'HoraDataDest'], right_on=['aero', 'HoraData'], how='left')

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
bimtra = bimtra.merge(stats_df, on="flightid", how="left")

# %%


# %%
bimtra

# %%
import pandas as pd

metaf = pd.read_csv("data/metaf.csv")
metaf['metaf'].apply(parse_metar)
metaf['metaf'] = metaf['metaf'].apply(lambda x: ' '.join(x.split()))

metaf[['Temperatura', 'Ponto de Orvalho', 'Velocidade do Vento', 'Direção do Vento', 'Visibilidade', 'Pressão', 'Tempo', 'Céu']] = metaf['metaf'].apply(parse_metar)

metaf['HoraData'] = pd.to_datetime(metaf['hora'], unit='ms').dt.strftime('%Y-%m-%d %H')
# Mesclar os dataframes usando a coluna "HoraData"


# %% [markdown]
# 

# %%

df_merge = pd.merge(bimtra, metaf, left_on=['destino', 'HoraDataDest'], right_on=['aero', 'HoraData'], how='left')
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

bimtra = bimtra.merge(stats_df, on="flightid", how="left")
# Preencher NaN na coluna de espera com 0 (sem espera)
bimtra

# %%
tcprev = pd.read_csv("data/tc-prev.csv")

# %%
tcprev["hora"] = pd.to_datetime(tcprev["hora"], unit='ms')

# %%
tcprev

# %%
tcprev.to_csv("data/tc-prev-join.csv", index=False)

# %%
bimtra.shape

# %%
# Mesclar os dataframes usando a coluna "HoraData"
tcprev['HoraData'] = tcprev['hora'].dt.strftime('%Y-%m-%d %H')
df_merge = pd.merge(bimtra, tcprev, on='HoraData', how='left')

# Preencher NaN na coluna de espera com 0 (sem espera)
df_merge['troca'] = df_merge['troca'].fillna(0)
df_merge = df_merge.groupby("flightid").agg({"troca": "sum"})
bimtra = bimtra.merge(df_merge.reset_index(), on="flightid", how="left")

# %%
bimtra.shape

# %%
tcprev["diaSemana"] = tcprev["hora"].dt.dayofweek

tcprev["hora"] = tcprev["hora"].dt.hour
tcprev_hora = tcprev.groupby(["aero", "hora"]).agg({"troca": ["mean", "sum"]})
tcprev_sem = tcprev.groupby(["aero", "diaSemana"]).agg({"troca": ["mean", "sum"]})
tcprev_aero = tcprev.groupby(["aero"]).agg({"troca": ["mean", "sum"]})

# %%


# %%
tcreal = pd.read_csv("data/tc-real.csv")
tcreal["hora"] = pd.to_datetime(tcreal["hora"], unit='ms')
tcreal["diaSemana"] = tcreal["hora"].dt.dayofweek

tcreal['HoraData'] = tcreal['hora'].dt.strftime('%Y-%m-%d %H')
tcreal["hora"] = tcreal["hora"].dt.hour

# %%
#tcreal = pd.read_csv("data/tc-real.csv")
tcreal

# %%
tcreal.to_csv("data/tc-real-join.csv", index=False)

# %%
# Mesclar os dataframes usando a coluna "HoraData"

df_merge = pd.merge(bimtra, tcreal, on='HoraData', how='left')

# Preencher NaN na coluna de espera com 0 (sem espera)
df_merge['trocareal'] = 1
df_merge = df_merge.groupby("flightid").agg({"trocareal": "sum"})
bimtra = bimtra.merge(df_merge.reset_index(), on="flightid", how="left")

# %%
bimtra

# %%

tcreal_hora = tcreal.groupby(["aero", "hora"]).agg({"aero": ["count"]})
tcreal_sem = tcreal.groupby(["aero", "diaSemana"]).agg({"aero": ["count"]})
tcreal_aero = tcreal.groupby(["aero"]).agg({"aero": ["count"]})

# %%
tcreal_aero

# %%
tcprev = pd.read_csv("data/tc-prev.csv")

# %%
bimtra["aero"] = bimtra["origem"].str[2:]

# %%
len(bimtra)

# %%
tcprev_hora.reset_index(inplace=True)

# %%
tcprev_hora.columns = tcprev_hora.columns.droplevel(0)

# %%
tcprev_hora.columns = ["aero", "hora", "troca_prev_mean", "troca_prev_sum"]
tcprev_aero.reset_index(inplace=True)
tcprev_aero.columns = tcprev_aero.columns.droplevel(0)
tcprev_aero.columns = ["aero", "troca_prev_mean", "troca_prev_sum"]
tcprev_sem.reset_index(inplace=True)
tcprev_sem.columns = tcprev_sem.columns.droplevel(0)
tcprev_sem.columns = ["aero", "diaSemana", "troca_prev_mean", "troca_prev_sum"]
tcreal_hora.reset_index(inplace=True)
tcreal_hora.columns = tcreal_hora.columns.droplevel(0)
tcreal_hora.columns = ["aero", "hora", "troca_real"]
tcreal_aero.reset_index(inplace=True)
tcreal_aero.columns = tcreal_aero.columns.droplevel(0)
tcreal_aero.columns = ["aero", "troca_real"]
tcreal_sem.reset_index(inplace=True)
tcreal_sem.columns = tcreal_sem.columns.droplevel(0)
tcreal_sem.columns = ["aero", "diaSemana", "troca_real"]


# %%


# %%
bimtra.columns

# %%
tcprev_hora.to_csv("data/tcprev_hora.csv", index=False)
tcprev_sem.to_csv("data/tcprev_sem.csv", index=False)
tcprev_aero.to_csv("data/tcprev_aero.csv", index=False)
tcreal_hora.to_csv("data/tcreal_hora.csv", index=False)
tcreal_sem.to_csv("data/tcreal_sem.csv", index=False)
tcreal_aero.to_csv("data/tcreal_aero.csv", index=False)

# %%
bimtra.shape

# %%
tcprev_sem

# %%
bimtra = bimtra.merge(tcprev_hora, on=["aero", "hora"], how="left")
bimtra = bimtra.merge(tcprev_sem, on=["aero", "diaSemana"], how="left")
bimtra = bimtra.merge(tcprev_aero, on=["aero"], how="left")
bimtra = bimtra.merge(tcreal_hora, on=["aero", "hora"], how="left")
bimtra = bimtra.merge(tcreal_sem, on=["aero", "diaSemana"], how="left")
bimtra = bimtra.merge(tcreal_aero, on=["aero"], how="left")
bimtra

# %%
bimtra["hora"]

# %%
train = bimtra.merge(resumo, right_on="linha_first", left_on="linha", how="left")

# %%
resumo

# %%
#esperas_semana.to_csv("data/esperas_semana.csv", index=False)
#esperas_hora.to_csv("data/esperas_hora.csv", index=False)

# %%
esperas

# %%
train["origem"], origem = pd.factorize(train["origem"]) #verificar se rodando novamente muda de código
train["destino"], destino = pd.factorize(train["destino"])


# %%


# %%
train["linha"], linha = pd.factorize(train["linha"])


# %%
pd.Series(origem).to_csv("data/origem.csv", index=False)
pd.Series(destino).to_csv("data/destino.csv", index=False)
pd.Series(linha).to_csv("data/linha.csv", index=False)

# %%
del train["linha_first"]


# %%
del train["dt_dep"], train["dt_arr"], train["aero_x"]

# %%
del train["aero_y"]

# %%
del train["aero"]

# %%
train

# %%
flightThrought = pd.read_csv("data/flightThourghtArea2.csv")

# %%
len(train)

# %%
train = train.merge(flightThrought, on="flightid", how="left")
train.columns = [re.sub('[\[\]<>\'"\'.,]', '', col) for col in train.columns]

# %%


# %%


train.to_csv("data/train_with_flight_v2.csv", index=False)

# %%
for col in train.columns:
    print(col)


