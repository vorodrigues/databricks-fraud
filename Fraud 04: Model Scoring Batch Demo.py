# Databricks notebook source
# MAGIC %md # Prep Data

# COMMAND ----------

# MAGIC %md On UC

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists hive_metastore.vr_fraud_dev.visits_silver as select * from vr_fraud.dev.visits_silver;
# MAGIC create table if not exists hive_metastore.vr_fraud_dev.customers_silver as select * from vr_fraud.dev.customers_silver;
# MAGIC create table if not exists hive_metastore.vr_fraud_dev.locations_silver as select * from vr_fraud.dev.locations_silver;

# COMMAND ----------

# MAGIC %md Off UC

# COMMAND ----------

# MAGIC %sql 
# MAGIC USE CATALOG hive_metastore;
# MAGIC USE DATABASE vr_fraud_dev

# COMMAND ----------

db = 'vr_fraud_dev'

# COMMAND ----------

import pyspark.pandas as ps

visits_df = spark.table(f"{db}.visits_silver").pandas_api()
customers_df = spark.table(f"{db}.customers_silver").pandas_api()
locations_df = spark.table(f"{db}.locations_silver").pandas_api()

# Step: Keep rows where amount > 0
visits_df = visits_df.loc[visits_df['amount'] > 0]

# Step: Drop missing values in [All columns]
visits_df = visits_df.dropna()

# Step: Drop duplicates based on ['amount', 'atm_id', 'customer_id', 'day', 'fraud_report', 'hour', 'min', 'month', 'sec', 'visit_id', 'withdrawl_or_deposit', 'year', 'date_visit']
visits_df = visits_df.drop_duplicates(keep='first')

# Step: Inner Join with customer_df where customer_id=customer_id
visits_df = ps.merge(visits_df, customers_df, how='inner', on=['customer_id'])

# Step: Inner Join with locations_df where atm_id=atm_id
visits_df = ps.merge(visits_df, locations_df, how='inner', on=['atm_id'])

# Step: Change data type of customer_since_date to Datetime
visits_df['customer_since_date'] = ps.to_datetime(visits_df['customer_since_date'], format='%Y-%m-%d')

# Step: Change data type of date_visit to Datetime
visits_df['date_visit'] = ps.to_datetime(visits_df['date_visit'], format='%Y-%m-%d')

# Step: Create new column 'customer_lifetime' from formula 'date_visit - customer_since_date'
visits_df['customer_lifetime'] = visits_df['date_visit'] - visits_df['customer_since_date']

# COMMAND ----------

visits_df = ps.sql('''
  SELECT
    visit_id,
    year,
    month,
    day,
    hour,
    min,
    amount,
    withdrawl_or_deposit,
    city_state_zip.state as state,
    pos_capability,
    offsite_or_onsite,
    bank,
    checking_savings,
    datediff(date_visit, customer_since_date) as customer_lifetime
  FROM {visits_df}
''', visits_df = visits_df)

# COMMAND ----------

df = visits_df.to_spark()

# COMMAND ----------

df.write.saveAsTable('fs_atm_visits')

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.write_table(
    name=db+".fs_atm_visits",
    df=df,
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %sql create table hive_metastore.vr_fraud_dev.train as select visit_id from hive_metastore.vr_fraud_dev.fs_atm_visits limit 1000000

# COMMAND ----------

# MAGIC %sql create table hive_metastore.vr_fraud_dev.test as select visit_id from hive_metastore.vr_fraud_dev.fs_atm_visits limit 1000000

# COMMAND ----------

# MAGIC %md # Environment Setup

# COMMAND ----------

# MAGIC %pip install category-encoders==2.5.1 psutil==5.8.0 typing-extensions==3.10.0.2 xgboost==1.6.2

# COMMAND ----------

# dbutils.widgets.text('db', 'vr_fraud_dev', 'Database')
# dbutils.widgets.text('path', '/FileStore/vr/fraud/dev', 'Path')

# COMMAND ----------

db = dbutils.widgets.get('db')
path = dbutils.widgets.get('path')
print('DATABASE: '+db)
print('PATH: '+path)

# COMMAND ----------

# MAGIC %sql USE DATABASE $db

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to use our trained model to generate predictions that may be imported into a downstream CRM system.  It should be run on a cluster leveraging Databricks ML 7.1+ and **CPU-based** nodes.

# COMMAND ----------

# MAGIC %md # Fraud 04: Model Scoring
# MAGIC
# MAGIC Finally, the model can be scored on a new dataset to get predictions.
# MAGIC
# MAGIC Models can be deployed as a batch, stream or real time proccess.<br><br>
# MAGIC
# MAGIC ![](/files/shared_uploads/victor.rodrigues@databricks.com/ml_4.jpg)

# COMMAND ----------

# MAGIC %md ## Step 1: Score Model

# COMMAND ----------

# DBTITLE 1,Create Feature Store Client
from databricks import feature_store
import os, shutil
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Generate Predictions with Production Model
model_name = 'VR Fraud Model'

preds = fs.score_batch(
    f'models:/{model_name}/Production',
    spark.table(db+'.test')
).selectExpr(
    "date( cast(year as string) || '-' || cast(month as string) || '-' || cast(day as string) ) as date",
    "timestamp( cast(year as string) || '-' || cast(month as string) || '-' || cast(day as string) || ' ' || cast(hour as string) || ':' || cast(min as string) ) as timestamp",
    '*'
)

display(preds)

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from (
# MAGIC   select rank() over (order by prediction desc) as rank, prediction from vr_fraud.dev.preds order by prediction desc 
# MAGIC )
# MAGIC where rank = 1000000 * 0.02

# COMMAND ----------

# MAGIC %md ## Step 2: Save Predictions
# MAGIC
# MAGIC Regardless of whether our intent is to use the [Microsoft Dynamics CRM Common Data Service](https://docs.microsoft.com/en-us/powerapps/developer/common-data-service/import-data) or [Salesforce DataLoader](https://developer.salesforce.com/docs/atlas.en-us.dataLoader.meta/dataLoader/data_loader.htm), we need to produce a UTF-8, delimited text file with a header row. We can deliver such a file as follows:

# COMMAND ----------

# MAGIC %md ### Step 2a: Delta Table

# COMMAND ----------

# DBTITLE 1,Save Predictions to a Delta Table
(
  preds
    .writeTo('preds').createOrReplace()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists vr_fraud.dev.preds as select * from hive_metastore.vr_fraud_dev.preds;
# MAGIC create table if not exists vr_fraud.dev.train as select * from hive_metastore.vr_fraud_dev.train;
# MAGIC create table if not exists vr_fraud.dev.test as select * from hive_metastore.vr_fraud_dev.test;
# MAGIC create table if not exists vr_fraud.dev.fs_atm_visits as select * from hive_metastore.vr_fraud_dev.fs_atm_visits;

# COMMAND ----------

# MAGIC %sql create or replace table vr_fraud.dev.preds_labels as select
# MAGIC   'VR Fraud Model' as model,
# MAGIC   p.*,
# MAGIC   case when p.prediction > 0.73 then 'Y' else 'N' end as fraud_prediction,
# MAGIC   l.fraud_report
# MAGIC from vr_fraud.dev.preds p
# MAGIC left join vr_fraud.dev.visits_gold l
# MAGIC on p.visit_id = l.visit_id

# COMMAND ----------

# MAGIC %sql create database vr_fraud.monitoring

# COMMAND ----------

# MAGIC %md ### Step 2b: CSV File

# COMMAND ----------

# DBTITLE 1,Save Predictions to a CSV File
output_path = path+'/output'

(preds
    .repartition(1)  # repartition to generate a single output file
    .write
    .mode('overwrite')
    .csv(
      path=output_path,
      sep=',',
      header=True,
      encoding='UTF-8'
      )
  )

# COMMAND ----------

# DBTITLE 1,Rename Output File
for file in os.listdir('/dbfs'+output_path):
  if file[-4:]=='.csv':
    shutil.move('/dbfs'+output_path+'/'+file, '/dbfs'+output_path+'/preds.csv' )

# COMMAND ----------

# DBTITLE 1,Examine Output File
print(dbutils.fs.head(output_path+'/preds.csv'))
