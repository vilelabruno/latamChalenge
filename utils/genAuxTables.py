from pyspark.sql import SparkSession
from pyspark.sql.functions import col, first, to_date, unix_timestamp
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Dataframe Example").getOrCreate()

cat62 = spark.read.csv('data/cat62.csv', header=True)

bimtra = spark.read.csv('data/bimtra.csv', header=True)
bimtra = bimtra.withColumn("linha", bimtra["origem"] + bimtra["destino"])
bimtra = bimtra.groupBy("flightid").agg(first("linha").alias("linha"))
bimtra = bimtra.select("*")
bimtra = bimtra.drop("linha")
cat62 = cat62.join(bimtra, "flightid", "left_outer")

normal_routes = spark.read.csv('data/normal_routes.csv', header=True)
cat62_filtered = cat62.join(normal_routes, cat62.flightid == normal_routes.flightid, how='inner')

test = spark.read.csv('data/idsc_dataset.csv', sep=';', header=True)
test = test.withColumn("linha_test", test["origem"] + test["destino"])

cat62_filtered = cat62_filtered.drop("flightid")

test = test.join(cat62_filtered, test.linha_test == cat62_filtered.linha, "left_outer")

test = test.drop('linha_test') 
test = test.withColumn("dt_radar", (col("dt_radar") / 1000).cast("timestamp"))
test = test.withColumn("dt_dep", to_date(unix_timestamp(col('dt_dep'), 'yyyy-MM-dd').cast("timestamp")))

test = test.orderBy("flightid", "dt_radar")

window = Window.partitionBy('flightid').orderBy('dt_radar')
first_route_event = test.select('flightid', first('dt_radar').over(window).alias('first_dt_radar'))

test = test.join(first_route_event, on='flightid', how="left")

test.write.csv('data/test_cat62_v2.csv', header=True)

test.show()
