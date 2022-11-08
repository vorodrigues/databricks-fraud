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

from pyspark.sql.functions import col, concat_ws, lpad, to_date

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

# MAGIC %md ## Examine the Data
# MAGIC There are several tables: customer data, customer ATM visits, and ATM locations; for later, a database of criminal mugshots!

# COMMAND ----------

display(dbutils.fs.ls(path+'/raw/atm_visits'))

# COMMAND ----------

# MAGIC %sql SELECT * FROM JSON.`$path/raw/atm_visits/part-00000-tid-6242017081815707263-37eaf674-661e-458e-a6d1-5a82e2198105-2-1-c000.json`

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
                .option("cloudFiles.maxFilesPerTrigger", 1) \
                .load(path+"/raw/atm_visits")

#Write Stream as Delta Table
bronzeDF.writeStream.format("delta") \
        .option("checkpointLocation", path+"/checkpoints/bronze") \
        .trigger(availableNow=True) \
        .toTable("visits_bronze")

# COMMAND ----------

display(spark.readStream.table('visits_bronze').groupBy('year', 'month', 'fraud_report').sum('amount').orderBy('year', 'month'))

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
silverDF = spark.readStream.table('visits_bronze') \
                .where('visit_id IS NOT NULL AND _rescued_data IS NULL') \
                .drop('_rescued_data') \
                .withColumn('date_visit', to_date(concat_ws('-', col('year'), lpad(col('month'),2,'0'), lpad(col('day'),2,'0')), 'yyyy-MM-dd')) \
                .where('date_visit IS NOT NULL')

#Write it back to your "turbine_silver" table
silverDF.writeStream.format('delta') \
        .option("checkpointLocation", path+"/checkpoints/silver") \
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
goldDF = spark.readStream.table('visits_silver') \
              .join(locations, on='atm_id', how='left') \
              .join(customers, on='customer_id', how='left') \
              .withColumn('state', col('city_state_zip.state')) \
              .withColumn('city', col('city_state_zip.city')) \
              .withColumn('zip', col('city_state_zip.zip')) \
              .drop('city_state_zip')

goldDF.writeStream.format('delta') \
      .option('checkpointLocation', path+'/checkpoints/gold') \
      .toTable('visits_gold')

# COMMAND ----------

# MAGIC %sql SELECT * FROM visits_gold LIMIT 100

# COMMAND ----------

# MAGIC %sql
# MAGIC select '1-bronze' as layer, count(*) as cnt from visits_bronze
# MAGIC union
# MAGIC select '2-silver' as layer, count(*) as cnt from visits_silver
# MAGIC union
# MAGIC select '3-gold' as layer, count(*) as cnt from visits_gold
# MAGIC order by layer

# COMMAND ----------

# MAGIC %md ### Constraints ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)
# MAGIC Let's create some constraints to avoid bad data to flow through our pipeline and to help us identify potential issues with our data.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE visits_gold CHANGE COLUMN visit_id SET NOT NULL;
# MAGIC ALTER TABLE visits_gold ADD CONSTRAINT tnxType CHECK (withdrawl_or_deposit IN ('deposit', 'withdrawl'));

# COMMAND ----------

# MAGIC %md Let's try to insert data with `null` id.

# COMMAND ----------

# MAGIC %sql INSERT INTO visits_gold VALUES (1, 1, 1, 1, 1, 1, 1, 1, 1, null, 'deposit', 1, null, '', '', '', 1, '', '', '', null, '', '', '')

# COMMAND ----------

# MAGIC %md Now, let's try to insert data with an invalid transaction type.

# COMMAND ----------

# MAGIC %sql INSERT INTO visits_gold VALUES (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 'transfer', 1, null, '', '', '', 1, '', '', '', null, '', '', '')

# COMMAND ----------

# MAGIC %md ### Schema Enforcement ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)
# MAGIC To show you how schema enforcement works, let's try to insert a record with a new column -- `new_col` -- that doesn't match our existing Delta Lake table schema.

# COMMAND ----------

# MAGIC %sql INSERT INTO visits_gold SELECT *, 0 as new_col FROM visits_gold LIMIT 1

# COMMAND ----------

# MAGIC %md **Schema enforcement helps keep our tables clean and tidy so that we can trust the data we have stored in Delta Lake.** The writes above were blocked because the schema of the new data did not match the schema of table (see the exception details). See more information about how it works [here](https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html).

# COMMAND ----------

# MAGIC %md ### Schema Evolution ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)
# MAGIC If we ***want*** to update our Delta Lake table to match this data source's schema, we can do so using schema evolution. Simply enable the `autoMerge` option.

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.schema.autoMerge.enabled = True;
# MAGIC INSERT INTO visits_gold SELECT *, 0 as new_col FROM visits_gold LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full DML Support ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)
# MAGIC 
# MAGIC Delta Lake brings ACID transactions and full DML support to data lakes: `DELETE`, `UPDATE`, `MERGE INTO`

# COMMAND ----------

# MAGIC %md
# MAGIC We just realized that something is wrong in the data before 2017! Let's DELETE all this data from our gold table as we don't want to have wrong value in our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM visits_gold where customer_since_date < '2016-02-01';

# COMMAND ----------

# MAGIC %md
# MAGIC With a legacy data pipeline, to insert or update a table, you must:
# MAGIC 1. Identify the new rows to be inserted
# MAGIC 2. Identify the rows that will be replaced (i.e. updated)
# MAGIC 3. Identify all of the rows that are not impacted by the insert or update
# MAGIC 4. Create a new temp based on all three insert statements
# MAGIC 5. Delete the original table (and all of those associated files)
# MAGIC 6. "Rename" the temp table back to the original table name
# MAGIC 7. Drop the temp table
# MAGIC 
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/merge-into-legacy.gif" alt='Merge process' width=600/>
# MAGIC 
# MAGIC 
# MAGIC #### INSERT or UPDATE with Delta Lake
# MAGIC 
# MAGIC 2-step process: 
# MAGIC 1. Identify rows to insert or update
# MAGIC 2. Use `MERGE`

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO visits_gold AS l
# MAGIC USING (SELECT * FROM visits_gold LIMIT 10) AS m
# MAGIC ON l.visit_id = m.visit_id
# MAGIC WHEN MATCHED THEN 
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED THEN
# MAGIC   INSERT *

# COMMAND ----------

# MAGIC %md ### Time Travel ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)
# MAGIC 
# MAGIC Delta Lake’s time travel capabilities simplify building data pipelines for use cases including:
# MAGIC 
# MAGIC * Auditing Data Changes
# MAGIC * Reproducing experiments & reports
# MAGIC * Rollbacks
# MAGIC 
# MAGIC As you write into a Delta table or directory, every operation is automatically versioned.
# MAGIC 
# MAGIC <img src="https://github.com/risan4841/img/blob/master/transactionallogs.png?raw=true" width=250/>
# MAGIC 
# MAGIC You can query snapshots of your tables by:
# MAGIC 1. **Version number**, or
# MAGIC 2. **Timestamp.**
# MAGIC 
# MAGIC using Python, Scala, and/or SQL syntax; for these examples we will use the SQL syntax.  
# MAGIC 
# MAGIC For more information, refer to the [docs](https://docs.delta.io/latest/delta-utility.html#history), or [Introducing Delta Time Travel for Large Scale Data Lakes](https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html)

# COMMAND ----------

# MAGIC %md Review Delta Lake Table History for  Auditing & Governance
# MAGIC 
# MAGIC All the transactions for this table are stored within this table including the initial set of insertions, update, delete, merge, and inserts with schema modification

# COMMAND ----------

# MAGIC %sql DESCRIBE HISTORY visits_gold

# COMMAND ----------

# MAGIC %md Use time travel to count records both in the latest version of the data, as well as the initial version.
# MAGIC 
# MAGIC As you can see, 10 new records was added.

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 'latest' as ver, count(*) as cnt FROM visits_gold
# MAGIC UNION
# MAGIC SELECT 'initial' as ver, count(*) as cnt FROM visits_gold VERSION AS OF 1

# COMMAND ----------

# MAGIC %md Rollback table to initial version using `RESTORE`

# COMMAND ----------

# MAGIC %sql RESTORE visits_gold VERSION AS OF 1
