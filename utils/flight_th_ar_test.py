# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

test = pd.read_csv("data/idsc_dataset.csv", delimiter=";")

# %%
test["data"] = pd.to_datetime(test["dt_dep"])

# %%
len(test["path"].unique())

# %%
import requests

def download_images(df):
    unlock = False
    for row in df["path"].unique():

        # Faz o download da imagem usando a biblioteca requests
        try:
            data = row.split('/')[-1]
            path = row
            
            # Extrai o nome do arquivo da URL
            nome_arquivo = path.split('/')[-1]
            print(nome_arquivo)
    
            # Define o caminho onde a imagem será salva
            pasta_destino = f'data/img/test/'
        
            response = requests.get(path)
        
            # Verifica se a resposta foi bem sucedida (código 200)
            if response.status_code == 200:
                with open(f'{pasta_destino}{data}.jpg', 'wb') as file:
                    file.write(response.content)
                    print(f"Imagem {nome_arquivo} salva em {pasta_destino}")
            else:
                print(f"Erro ao baixar a imagem {nome_arquivo}")
        except:
            pass


# %%
download_images(test)

# %%
import os
import subprocess
import pandas as pd
import json
from multiprocessing import Pool, cpu_count

# Function to be mapped
def process_file(file):
    log = []
    data_list = []
    file_path = os.path.join('data/img/test/', file)
    print(file)
    try:
        output = subprocess.check_output(['python3', 'utils/decodeImage.py', file_path])
        output = json.loads(output)
        
        polyCount = 0
        for poly in output:
            verticeCount = 0
            for vertice in poly:
                x = vertice[0][0]
                y = vertice[0][1]
                data_list.append({'data': file, 'X': x, 'Y': y, 'poly': polyCount, 'vertice': verticeCount})
                verticeCount = verticeCount + 1
            polyCount = polyCount + 1
        log.append(f'{file}: sem erros\n')
    except subprocess.CalledProcessError as err:
        log.append(f'{file}: erro na execução - {err}\n')
    
    return data_list, log


# Use Pool to create subprocesses
with Pool(min(8, cpu_count())) as p:
    results = p.map(process_file, os.listdir('data/img/test/'))

# Combine results
all_data = []
all_logs = []
for data, log in results:
    all_data.extend(data)
    all_logs.extend(log)

# Creates dataframe from the data
dataset = pd.DataFrame(all_data)

# Cria o arquivo de log
with open('log/log2.txt', 'w') as log_file:
    log_file.writelines(all_logs)

# Export dataframe to csv
dataset.to_csv('data/testImageData.csv', index=False)


# %%
import pandas as pd
dataset = pd.read_csv('data/testImageData.csv')

# %%
import pandas as pd
from shapely.geometry import Polygon

#dataset['X'] = dataset["X"].apply(lambda x: x+940)
#dataset['Y'] = dataset["Y"].apply(lambda x: x+740)
polygons = dataset.groupby(['data', 'poly'])[['X', 'Y']].apply(lambda x: Polygon(x.values))

# Calculate the centroids
df_centroids = polygons.apply(lambda polygon: polygon.centroid)
#
## Estimate the area
df_area = polygons.apply(lambda polygon: polygon.area)
#
## Create a DataFrame with the results
results = pd.DataFrame({
    'polygon': polygons.index,
    'centroid_x': [p.x for p in df_centroids],  
    'centroid_y': [p.y for p in df_centroids], 
    'area_estimate': df_area.values
})


# %%
results

# %%
results["poly"] = results["polygon"].apply(lambda x: str(x).split(",")[1].split(")")[0].replace(" ","").replace("(","").replace("'",""))
results["data"] = pd.to_datetime(results["polygon"].apply(lambda x: str(x).split(",")[0].split("_")[1].split(".")[0]))

# %%
results["centroid_x"] = results["centroid_x"].apply(lambda x: x + 740)
results["centroid_y"] = results["centroid_y"].apply(lambda x: x + 940)

# %%


import numpy as np
from scipy import interpolate

def create_transform(pixel_coords, latlon_coords):

    pixel_x, pixel_y = zip(*pixel_coords)
    lat, lon = zip(*latlon_coords)

    lat_interp = interpolate.interp1d(pixel_y, lat, fill_value="extrapolate")
    lon_interp = interpolate.interp1d(pixel_x, lon, fill_value="extrapolate")

    return lat_interp, lon_interp

def convert_pixel_to_latlong(x, y):
    lat_lon_points = [(7.455477, -77.829943), (-51.848798, -76.538633), (1.355664, -28.966528), (8.000000, -29.500000)]
    x_y_points = [(940, 740), (940, 2192), (2092, 2192), (2092, 740)]
    lat_interp, lon_interp = create_transform(x_y_points, lat_lon_points)
    return lat_interp(y), lon_interp(x)


# Para converter um ponto de pixel para lat/lon:
lat, lon = convert_pixel_to_latlong(1500, 1500) # -3.152260, -40.788782
print(lat, lon)

results["lat"] = results.apply(lambda x: convert_pixel_to_latlong(x["centroid_x"], x["centroid_y"])[1], axis=1)
results["lon"] = results.apply(lambda x: convert_pixel_to_latlong(x["centroid_x"], x["centroid_y"])[0], axis=1)
#dataset["lat"] = dataset.apply(lambda x: convert_pixel_to_latlong(x["X"], x["Y"])[1], axis=1)
#dataset["lon"] = dataset.apply(lambda x: convert_pixel_to_latlong(x["X"], x["Y"])[0], axis=1)

# %%
results

# %%
results.to_csv("data/test_storm_areas", index=False)

# %%
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .appName('my-cool-app') \
    .getOrCreate()
spark.conf.set('spark.sql.crossJoin.enabled', 'true') 




# %%

# Assuming 'df_areas' is your first dataframe with centroid and 'df_flight' is your second dataframe
df_areas = spark.read.csv('./data/test_storm_areas', inferSchema=True, header=True)
# Register your dataframes as temp views to use them in SQL
df_areas.createOrReplaceTempView('df_areas')

# %%
df_flight = spark.read.csv('data/test_cat62.csv', inferSchema=True, header=True)

df_flight.createOrReplaceTempView('df_flight')


# %%
import numpy as np
import pandas as pd
df_flight = pd.read_csv('data/test_cat62.csv')
df_areas = pd.read_csv('./data/test_storm_areas')
# Register your dataframes as temp views to use them in SQL
# Assuming df_flight and df_areas are your dataframes
from math import sin, cos, sqrt, asin, radians

def haversine(row):
     # The radius of the earth in km
    R = 6371 
    lat1 = radians(row['f_lat'])
    lat2 = radians(row['a_lat'])
    dlat = lat2 - lat1
    dlon = radians(row['a_lon'] - row['f_lon'])
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# Assuming df_flight and df_areas are your dataframes
df_flight['rounded_event_time'] = df_flight['data'].apply(lambda x: str(x)[:13])
df_areas['data'] = df_areas['data'].apply(lambda x: str(x)[:13])


# %%


# %%
df_flight = df_flight[['rounded_event_time', 'lat', 'lon', 'flightid']]
df_areas = df_areas[['data', 'lat', 'lon', 'area_estimate']]

# %%

# Join the dataframes on rounded_event_time
merged_df = pd.merge(df_flight, df_areas, left_on='rounded_event_time', right_on='data', 
                     suffixes=('_f', '_a'), how='left')

# %%


# Create 'flight_through_area' column
merged_df['flight_through_area'] = np.where(haversine(merged_df) <= merged_df['area_estimate'] / 2, 1, 0)

#Group by flightid
grouped_df = merged_df.groupby(['flightid'])

#Get the first of polygon and sum of flight_through_area
result_df = grouped_df.agg({'polygon' : lambda x: str(x.values[0])[2:13], 
                            'flight_through_area' : 'sum'}).reset_index()




# %%
from pyspark.sql.functions import *
#df_flight = df_flight.withColumn("dt_radar", to_timestamp(df_flight["dt_radar"] / 1000))

# %%

df_flight.createOrReplaceTempView('df_flight')

# %%
df_areas.show()

# %%
df_flight.show()

# %%
df_flight = df_flight.select('data', 'lat', 'lon', 'flightid')
df_areas = df_areas.select('data', 'lat', 'lon', 'area_estimate')


# %%


# %%
# Now run the SQL Query to join these two dataframes based on your conditions
query = """
    SELECT 
        f.flightid, first(SUBSTRING(CAST(a.polygon AS STRING), 3, 13)), 
    sum(CASE
            WHEN (2 * 6371 * ASIN(SQRT(POW(SIN((f.lat - a.lat) * .0174532925 / 2), 2) 
                                   + COS(f.lat * .0174532925) * COS(a.lat * .0174532925) *
                                   POW(SIN((f.lon - a.lon) * .0174532925 / 2), 2))))
                <= (a.area_estimate / 2) THEN 1
            ELSE 0
        END) as flight_through_area
    FROM (
        SELECT *,data,
        SUBSTRING(CAST(data AS STRING), 1, 13) AS rounded_event_time
        FROM df_flight
    ) f
    LEFT JOIN df_areas a
    ON f.data = a.data
    GROUP BY f.flightid
"""
result_df = spark.sql(query)

result_df.show()


# %%
result_df.repartition(1).write.format('csv').option('header',True).save('test_flight_area.csv')

# %%



