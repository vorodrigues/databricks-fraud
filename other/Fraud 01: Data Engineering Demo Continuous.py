# Databricks notebook source
# dbutils.widgets.text('db', 'vr_fraud_dev', 'Database')
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

from pyspark.sql.functions import col, concat_ws, lpad, to_date, lit

# COMMAND ----------

tables = ["visits_bronze", "visits_silver", "visits_gold"]
for table in tables:
  spark.sql("DROP TABLE IF EXISTS "+table)
  dbutils.fs.rm("/user/hive/warehouse/"+db+".db/"+table, True)

# COMMAND ----------

dbutils.fs.rm(path+"/checkpoints", True)

# COMMAND ----------

# DBTITLE 0,Overview
# MAGIC %md # ATM Fraud Analytics
# MAGIC 
# MAGIC **According to the Secret Service, the crime is responsible for about $350,000 of monetary losses each day in the United States and is considered to be the number one ATM-related crime. Trade group Global ATM Security Alliance estimates that skimming costs the U.S.-banking industry about $60 million a year.**
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

# MAGIC %md # Fraud 01: Data Engineering
# MAGIC 
# MAGIC The first step to prevent **ATM fraud** is to make sure we can ingest, clean and organize our data in a **performant**, **reliable** and **cost efficient** manner.
# MAGIC 
# MAGIC Here we are going to create a streaming pipeline to increasingly improve our data quality while moving through different layers of our **Lakehouse** and deliver it to be consumed by **BI** reports and **ML** models.
# MAGIC 
# MAGIC **Delta Lake** is a key enabler of this architecture, reducing the work required by data engineers to develop and maintain these pipelines.<br><br>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/09/delta-lake-medallion-model-scaled.jpg" width=1012/>

# COMMAND ----------

# MAGIC %md ## 2/ Silver layer: cleanup data and remove unecessary column

# COMMAND ----------

import datetime

def processedTime(microBatchDF, batchId):
  microBatchDF \
    .withColumn('processedTime', lit(datetime.datetime.now())) \
    .write \
    .option("txnAppId", "processedTime") \
    .option("txnVersion", batchId) \
    .mode("append") \
    .saveAsTable("visits_silver")

# COMMAND ----------

#Cleanup the silver table
#Our bronze table might have some malformed records.
#Filter on _rescued_data to select only the rows without json error, filter on visit_id not null and drop unnecessary columns
silverDF = spark.readStream.table('visits_bronze') \
                .where('visit_id IS NOT NULL AND _rescued_data IS NULL') \
                .drop('_rescued_data') \
                .withColumn('date_visit', to_date(concat_ws('-', col('year'), lpad(col('month'),2,'0'), lpad(col('day'),2,'0')), 'yyyy-MM-dd')) \
                .where('date_visit IS NOT NULL')

#Write it back to your "turbine_silver" table
silverDF.writeStream.format('delta') \
        .foreachBatch(processedTime) \
        .option("checkpointLocation", path+"/checkpoints/silver") \
        .outputMode("update") \
        .trigger(processingTime="1 second") \
        .start()

# COMMAND ----------

# MAGIC %md ## 3/ Gold layer: join location and customer information

# COMMAND ----------

locations = spark.table('locations_silver')
customers = spark.table('customers_silver')

#Join location and customer information
goldDF = spark.readStream.table('visits_silver') \
              .join(locations, on='atm_id', how='left') \
              .join(customers, on='customer_id', how='left') \
              .withColumn('state', col('city_state_zip.state')) \
              .withColumn('city', col('city_state_zip.city')) \
              .withColumn('zip', col('city_state_zip.zip')) \
              .drop('city_state_zip')

goldDF.writeStream.format('delta') \
      .option("mergeSchema", "true") \
      .option('checkpointLocation', path+'/checkpoints/gold') \
      .toTable('visits_gold')

# COMMAND ----------

# MAGIC %md ## 1/ Bronze layer: ingest data stream

# COMMAND ----------

schema = spark.read.option("inferSchema", True).json(path+"/raw/atm_visits/part-00000-tid-3679982078169745009-ddf735d2-4ace-4422-9941-2006de988611-7-1-c000.json").schema

# COMMAND ----------

dbutils.fs.rm(path+"/checkpoints/bronze", True)

# COMMAND ----------

while True:
  
  #Ingest data using Auto Loader.
  bronzeDF = spark.readStream \
                  .schema(schema) \
                  .option("maxFilesPerTrigger", 1) \
                  .json(path+"/raw/atm_visits")

  #Write Stream as Delta Table
  bronzeDF.writeStream.format("delta") \
          .option("checkpointLocation", path+"/checkpoints/bronze") \
          .trigger(processingTime="1 second") \
          .toTable("visits_bronze") \
          .awaitTermination(300)

  #Remove checkpoint
  dbutils.fs.rm(path+"/checkpoints/bronze", True)

# COMMAND ----------

# MAGIC %sql select count(*) from visits_gold
