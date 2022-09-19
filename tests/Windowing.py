# Databricks notebook source
# dbutils.widgets.text('path', '/FileStore/vr/fraud/dev', 'Path')

# COMMAND ----------

path = dbutils.widgets.get('path')
print('PATH: '+path)

# COMMAND ----------

dbutils.fs.rm(path+"/checkpoints", True)

# COMMAND ----------

spark.conf.set('spark.sql.shuffle.partitions', '32')
spark.conf.set("spark.sql.streaming.stateStore.providerClass", "com.databricks.sql.streaming.state.RocksDBStateStoreProvider")
spark.conf.set("spark.databricks.streaming.statefulOperator.asyncCheckpoint.enabled", "true")

# COMMAND ----------

bronzeDF = spark.readStream \
  .option('maxBytesPerTrigger', '10MB') \
  .table("vr_fraud_dev.visits_bronze")

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

bronzeDF = bronzeDF.withColumn('ts', current_timestamp())

# COMMAND ----------

from pyspark.sql.functions import window, col, sum, when, expr

bronzeDF = (bronzeDF.groupBy('visit_id', window('ts', '10 seconds', '5 seconds'))
  .agg(
    sum(when(col('ts') <= col('window.end') - expr('INTERVAL 5 seconds'), col('amount')).otherwise(0)).alias('amt5'),
    sum('amount').alias('amt10')
  )
  .select(
    'visit_id',
    col('window.end').alias('ts'),
    'amt5',
    'amt10'
  )
)

# COMMAND ----------

bronzeDF.writeStream \
  .format('noop') \
  .outputMode('update') \
  .start()
#   .trigger(processingTime='5 seconds') \

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# dbutils.fs.rm(path+"/checkpoints", True)
# spark.sql('drop table vr_fraud_dev.visits_bronze')

# spark.conf.set('spark.sql.shuffle.partitions', '32')
# spark.conf.set("spark.sql.streaming.stateStore.providerClass", "com.databricks.sql.streaming.state.RocksDBStateStoreProvider")
# spark.conf.set("spark.databricks.streaming.statefulOperator.asyncCheckpoint.enabled", "false")

# bronzeDF = spark.readStream.format("cloudFiles") \
#   .option("cloudFiles.format", "json") \
#   .option("cloudFiles.schemaLocation", path+"/schemas") \
#   .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
#   .option("cloudFiles.inferColumnTypes", True) \
#   .load(path+"/raw/atm_visits")

# bronzeDF.writeStream.format('delta') \
#   .option("checkpointLocation", path+"/checkpoints/bronze") \
#   .trigger(once=True) \
#   .toTable("vr_fraud_dev.visits_bronze") \
#   .processAllAvailable()

# COMMAND ----------

# %sql select count(*) from vr_fraud_dev.visits_bronze

# COMMAND ----------

# bronzeDF = spark.readStream.format("cloudFiles") \
#   .option("cloudFiles.format", "json") \
#   .option("cloudFiles.schemaLocation", path+"/schemas") \
#   .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
#   .option("cloudFiles.inferColumnTypes", True) \
#   .option("cloudFiles.maxBytesPerTrigger", "1MB") \
#   .load(path+"/raw/atm_visits")

# COMMAND ----------

# from pyspark.sql.window import Window
# from pyspark.sql.functions import window, col, sum, max, first

# w2 = Window.orderBy('ts').rangeBetween(-2, Window.currentRow)
# w5 = Window.orderBy('ts').rangeBetween(-5, Window.currentRow)

# # bronzeDF.groupBy(window('ts', '5 seconds', '1 seconds'), col('visit_id')) \
# bronzeDF = bronzeDF.groupBy('visit_id', 'ts', 'amount') \
#   .agg(
#     first('ts'),
#     sum('amount').over(w2).alias('amt2'),
#     sum('amount').over(w5).alias('amt5')
#   ) \
#   .select(
#     col('visit_id'),
#     col('ts'),
#     col('amt2'),
#     col('amt5')
#   )
