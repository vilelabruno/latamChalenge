
import numpy as np
from scipy import interpolate

from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import pandas as pd
from shapely.geometry import Polygon
import math
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .appName('my-cool-app') \
    .getOrCreate()
spark.conf.set('spark.sql.crossJoin.enabled', 'true') 




dataset = pd.read_csv('data/imageData.csv')

# %%
cat62 = pd.read_csv('data/cat-62.csv')

# %%
cat62["dt_radar"] = pd.to_datetime(cat62["dt_radar"], unit='ms')

# %%
cat62.sort_values(by=['dt_radar'], inplace=True)

#apply math degress to lat lon columns
cat62["lat"] = cat62["lat"].apply(math.degrees)
cat62["lon"] = cat62["lon"].apply(math.degrees)


dataset['X'] = dataset["X"].apply(lambda x: x+940)
dataset['Y'] = dataset["Y"].apply(lambda x: x+740)
polygons = dataset.groupby(['data', 'poly'])[['X', 'Y']].apply(lambda x: Polygon(x.values))

# Calculate the centroids
df_centroids = polygons.apply(lambda polygon: polygon.centroid)

# Estimate the area
df_area = polygons.apply(lambda polygon: polygon.area)

# Create a DataFrame with the results
results = pd.DataFrame({
    'polygon': polygons.index,
    'centroid_x': [p.x for p in df_centroids],  
    'centroid_y': [p.y for p in df_centroids], 
    'area_estimate': df_area.values
})


# %%
results["poly"] = results["polygon"].apply(lambda x: str(x).split(",")[1].split(")")[0].replace(" ","").replace("(","").replace("'",""))
results["data"] = pd.to_datetime(results["polygon"].apply(lambda x: str(x).split(",")[0].split("'")[1].split(".")[0]))

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

# %%
results.to_csv("data/storm_areas.csv", index=False)
# Now run the SQL Query to join these two dataframes based on your conditions
df_areas = spark.read.csv('data/storm_areas.csv', inferSchema=True, header=True)
# Register your dataframes as temp views to use them in SQL
df_areas.createOrReplaceTempView('df_areas')
df_flight = spark.read.csv('data/cat-62.csv', inferSchema=True, header=True)

df_flight.createOrReplaceTempView('df_flight')

df_flight = df_flight.withColumn("dt_radar", to_timestamp(df_flight["dt_radar"] / 1000))

df_flight.createOrReplaceTempView('df_flight')
query = """
    SELECT 
        f.flightid, f.dt_radar,
    MAX(CASE
            WHEN (2 * 6371 * ASIN(SQRT(POW(SIN((f.lat - a.lat) * .0174532925 / 2), 2) 
                                   + COS(f.lat * .0174532925) * COS(a.lat * .0174532925) *
                                   POW(SIN((f.lon - a.lon) * .0174532925 / 2), 2))))
                <= (a.area_estimate / 2) THEN 1
            ELSE 0
        END) as flight_through_area
    FROM (
        SELECT *,
        SUBSTRING(CAST(dt_radar AS STRING), 1, 13) AS rounded_event_time
        FROM df_flight
    ) f
    LEFT JOIN df_areas a
    ON f.rounded_event_time = SUBSTRING(CAST(a.polygon AS STRING), 3, 13)
    GROUP BY f.flightid, f.dt_radar
"""
result_df = spark.sql(query)

result_df.show()

result_df.write.csv("data/flightThourghtArea2.csv", header=True, mode="overwrite")