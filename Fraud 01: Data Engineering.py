# Databricks notebook source
# dbutils.widgets.text('db', 'vr_fraud_dev', 'Databse')
# dbutils.widgets.text('path', '/FileStore/vr/fraud/dev', 'Path')

# COMMAND ----------

db = dbutils.widgets.get('db')
path = dbutils.widgets.get('path')
print('DATABASE: '+db)
print('PATH: '+path)

# COMMAND ----------

# MAGIC %sql USE $db

# COMMAND ----------

# MAGIC %sql SET spark.databricks.delta.schema.autoMerge.enabled = False

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, lpad, to_date

# COMMAND ----------

tables = ["visits_bronze", "visits_silver", "visits_gold"]
for table in tables:
  spark.sql("DROP TABLE IF EXISTS "+table)
  dbutils.fs.rm("/user/hive/warehouse/"+db+".db/"+table, True)

# COMMAND ----------

dbutils.fs.rm(path+"/checkpoints", True)

# COMMAND ----------

# MAGIC %md # Fraud 01: Data Engineering

# COMMAND ----------

# DBTITLE 0,Overview
# MAGIC %md
# MAGIC 
# MAGIC *** According to the Secret Service, the crime is responsible for about $350,000 of monetary losses each day in the United States and is considered to be the number one ATM-related crime. Trade group Global ATM Security Alliance estimates that skimming costs the U.S.-banking industry about $60 million a year.***
# MAGIC 
# MAGIC The easiest way that companies identify atm fraud is by recognizing a break in spending patterns.  For example, if you live in Wichita, KS and suddenly your card is used to buy something in Bend, OR – that may tip the scales in favor of possible fraud, and your credit card company might decline the charges and ask you to verify them.
# MAGIC 
# MAGIC  
# MAGIC ![Databricks for Credit Card Fraud](https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/fighting_atm_fraud/credit_card_with_padlock.jpg)  
# MAGIC * [**ATM Fraud Analytics**](https://www.csoonline.com/article/2124891/fraud-prevention/atm-skimming--how-to-recognize-card-fraud.html) is the use of data analytics and machine learning to detect ATM fraud and is...  
# MAGIC   * Built on top of Databricks Platform
# MAGIC   * Uses a machine learning implementation to detect ATM fraud   
# MAGIC * This demo...  
# MAGIC   * demonstrates a ATM fraud detection workflow.  We use dataset is internally mocked up data.

# COMMAND ----------

# MAGIC %md ## Examine the Data
# MAGIC There are several tables: customer data, customer ATM visits, and ATM locations; for later, a database of criminal mugshots!

# COMMAND ----------

display(dbutils.fs.ls(path+'/raw/atm_visits'))

# COMMAND ----------

# MAGIC %sql SELECT * FROM JSON.`$path/raw/atm_visits`

# COMMAND ----------

# MAGIC %md ## 1/ Bronze layer: ingest data stream

# COMMAND ----------

# MAGIC %md ### Auto Loader
# MAGIC 
# MAGIC Auto Loader incrementally and efficiently processes new data files as they arrive in cloud storage.
# MAGIC 
# MAGIC Auto Loader provides a Structured Streaming source called cloudFiles. Given an input directory path on the cloud file storage, the cloudFiles source automatically processes new files as they arrive, with the option of also processing existing files in that directory.
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2020/02/autoloader.png)

# COMMAND ----------

#Ingest data using Auto Loader.
bronzeDF = spark.readStream.format("cloudFiles") \
                .option("cloudFiles.format", "json") \
                .option("cloudFiles.schemaLocation", path+"/schemas") \
                .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
                .option("cloudFiles.inferColumnTypes", True) \
                .load(path+"/raw/atm_visits")

# Write Stream as Delta Table
bronzeDF.writeStream.format("delta") \
        .option("checkpointLocation", path+"/checkpoints/bronze") \
        .trigger(once=True) \
        .toTable("visits_bronze")

# COMMAND ----------

# MAGIC %sql SELECT * FROM visits_bronze LIMIT 100

# COMMAND ----------

# MAGIC %md ### Auto Optimize
# MAGIC 
# MAGIC Auto Optimize is an optional set of features that automatically compact small files during individual writes to a Delta table. Paying a small cost during writes offers significant benefits for tables that are queried actively. It is particularly useful in the following scenarios:<br><br>
# MAGIC   - Streaming use cases where latency in the order of minutes is acceptable
# MAGIC   - MERGE INTO is the preferred method of writing into Delta Lake
# MAGIC   - CREATE TABLE AS SELECT or INSERT INTO are commonly used operations
# MAGIC ![](https://docs.databricks.com/_images/optimized-writes.png)

# COMMAND ----------

# MAGIC %sql ALTER TABLE visits_bronze SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true)

# COMMAND ----------

# MAGIC %md ## 2/ Silver layer: cleanup data and remove unecessary column

# COMMAND ----------

# MAGIC %md ### Cleanup data

# COMMAND ----------

#Cleanup the silver table
#Our bronze table might have some malformed records.
#Filter on _rescued_data to select only the rows without json error, filter on visit_id not null and drop unnecessary columns
silverDF = bronzeDF  \
                .where('visit_id IS NOT NULL AND _rescued_data IS NULL')\
                .drop('_rescued_data') \
                .withColumn('date_visit', to_date(concat_ws('-', col('year'), lpad(col('month'),2,'0'), lpad(col('day'),2,'0')), 'yyyy-MM-dd')) \
                .where('date_visit IS NOT NULL')

#Write it back to your "turbine_silver" table
silverDF.writeStream.format('delta') \
        .option("checkpointLocation", path+"/checkpoints/silver") \
        .trigger(once=True) \
        .toTable("visits_silver")

# COMMAND ----------

# MAGIC %sql SELECT * FROM visits_silver LIMIT 100

# COMMAND ----------

# MAGIC %md ## 3/ Gold layer: join location and customer information

# COMMAND ----------

# MAGIC %md ### Join batch data

# COMMAND ----------

locations = spark.table('locations_silver')
customers = spark.table('customers_silver')

#Join location and customer information
silverDF.join(locations, on='atm_id', how='left') \
        .join(customers, on='customer_id', how='left') \
        .writeStream.format('delta') \
        .option('checkpointLocation', path+'/checkpoints/gold') \
        .trigger(once=True) \
        .toTable('visits_gold')

# COMMAND ----------

# MAGIC %sql SELECT * FROM visits_gold LIMIT 100

# COMMAND ----------

# MAGIC %md ## Stop all active streams

# COMMAND ----------

for s in spark.streams.active:
  s.stop()
