# Databricks notebook source
from pyspark.sql.functions import when, col

# COMMAND ----------

# MAGIC %md # Dev Tables

# COMMAND ----------

# MAGIC %md ## Database

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS vr_fraud_dev

# COMMAND ----------

# MAGIC %md ## atm_visits.json

# COMMAND ----------

spark.read.parquet('/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_visits') \
     .withColumn('visit_id', when((col('visit_id') == 42701999) & (col('customer_id') == 204919), 44932650).otherwise(col('visit_id'))) \
     .write.json('/FileStore/vr/fraud/dev/raw/atm_visits')

# COMMAND ----------

# MAGIC %md ## customers_silver

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_dev.customers_silver AS SELECT * FROM parquet.`/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_customers`

# COMMAND ----------

# MAGIC %md ## locations_silver

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_dev.locations_silver AS SELECT * FROM parquet.`/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_locations`

# COMMAND ----------

# MAGIC %md ## train

# COMMAND ----------

spark.read.parquet('/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_visits') \
     .selectExpr("case when visit_id = 42701999 and customer_id = 204919 then 44932650 else visit_id end as visit_id", 
                 "case fraud_report when 'Y' then 1 when 'N' then 0 else null end as fraud_report") \
     .limit(10000) \
     .writeTo('vr_fraud_dev.train').createOrReplace()

# COMMAND ----------

# MAGIC %md ## test

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_dev.test AS SELECT visit_id FROM vr_fraud_dev.train

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Feature Store
# MAGIC <br>
# MAGIC 
# MAGIC - Use notebook 1 to generate table visits_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW vw_atm_visits AS SELECT
# MAGIC   visit_id,
# MAGIC   year,
# MAGIC   month,
# MAGIC   day,
# MAGIC   hour,
# MAGIC   min,
# MAGIC   amount,
# MAGIC   withdrawl_or_deposit,
# MAGIC   city_state_zip.state as state,
# MAGIC   pos_capability,
# MAGIC   offsite_or_onsite,
# MAGIC   bank,
# MAGIC   checking_savings,
# MAGIC   datediff('2018-01-01', customer_since_date) as customer_lifetime
# MAGIC FROM vr_fraud_dev.visits_gold

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists vr_fraud_dev.fs_atm_visits;
# MAGIC drop table if exists vr_fraud_test.fs_atm_visits;
# MAGIC drop table if exists vr_fraud_prod.fs_atm_visits;

# COMMAND ----------

fs.create_table(
    name="vr_fraud_dev.fs_atm_visits",
    df=spark.table("vw_atm_visits").limit(0),
    primary_keys=["visit_id"],
    description="ATM Fraud features [DEV]"
)

# COMMAND ----------

fs.create_table(
    name="vr_fraud_test.fs_atm_visits",
    df=spark.table("vw_atm_visits").limit(0),
    primary_keys=["visit_id"],
    description="ATM Fraud features [TEST]"
)

# COMMAND ----------

fs.create_table(
    name="vr_fraud_prod.fs_atm_visits",
    df=spark.table("vw_atm_visits").limit(0),
    primary_keys=["visit_id"],
    description="ATM Fraud features [PROD]"
)

# COMMAND ----------

# MAGIC %md ## AutoML 
# MAGIC <br>
# MAGIC 
# MAGIC - Use notebook 2 to generate feature table

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import FeatureLookup

fs = feature_store.FeatureStoreClient()

feature_lookups = [
    FeatureLookup(
      table_name = 'vr_fraud_dev.fs_atm_visits',
      feature_names = None,
      lookup_key = ['visit_id']
    )
]

training_set = fs.create_training_set(
  df = spark.table('vr_fraud_dev.train'),
  feature_lookups = feature_lookups,
  label = 'fraud_report',
  exclude_columns = ['visit_id']
)
df = training_set.load_df()

df.writeTo('vr_fraud_dev.train_dataset').createOrReplace()

# COMMAND ----------

# MAGIC %md # Test Tables

# COMMAND ----------

# MAGIC %md ## Database

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS vr_fraud_test

# COMMAND ----------

# MAGIC %md ## atm_visits.json

# COMMAND ----------

spark.read.json('/FileStore/vr/fraud/dev/raw/atm_visits') \
     .limit(10) \
     .write.json('/FileStore/vr/fraud/test/raw/atm_visits')

# COMMAND ----------

# MAGIC %md ## customers_silver

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE vr_fraud_test.customers_silver AS SELECT
# MAGIC   a.* 
# MAGIC FROM vr_fraud_dev.customers_silver AS a
# MAGIC RIGHT JOIN json.`/FileStore/vr/fraud/test/raw/atm_visits` AS b
# MAGIC ON a.customer_id = b.customer_id

# COMMAND ----------

# MAGIC %md ## locations_silver

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE vr_fraud_test.locations_silver AS SELECT
# MAGIC   a.* 
# MAGIC FROM vr_fraud_dev.locations_silver AS a
# MAGIC RIGHT JOIN json.`/FileStore/vr/fraud/test/raw/atm_visits` AS b
# MAGIC ON a.atm_id = b.atm_id

# COMMAND ----------

# MAGIC %md ## test

# COMMAND ----------

spark.read.json('/FileStore/vr/fraud/test/raw/atm_visits') \
     .select('visit_id') \
     .writeTo('vr_fraud_test.test').createOrReplace()

# COMMAND ----------

# MAGIC %md # Prod Tables

# COMMAND ----------

# MAGIC %md ## Database

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS vr_fraud_prod

# COMMAND ----------

# MAGIC %md ## atm_visits.json

# COMMAND ----------

spark.read.json('/FileStore/vr/fraud/dev/raw/atm_visits') \
     .write.json('/FileStore/vr/fraud/prod/raw/atm_visits')

# COMMAND ----------

# MAGIC %md ## customers_silver

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_prod.customers_silver AS SELECT * FROM vr_fraud_dev.customers_silver

# COMMAND ----------

# MAGIC %md ## locations_silver

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_prod.locations_silver AS SELECT * FROM vr_fraud_dev.locations_silver

# COMMAND ----------

# MAGIC %md ## test

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_prod.test AS SELECT * FROM vr_fraud_dev.test

# COMMAND ----------

# MAGIC %md # Post-Run Validation

# COMMAND ----------

# MAGIC %md ## 01: Data Engineering

# COMMAND ----------

# MAGIC %sql
# MAGIC select 'raw' as layer, count(*) as cnt from JSON.`/FileStore/vr/fraud/dev/raw/atm_visits`
# MAGIC union
# MAGIC select 'bronze' as layer, count(*) as cnt from vr_fraud_dev.visits_bronze
# MAGIC union
# MAGIC select 'silver' as layer, count(*) as cnt from vr_fraud_dev.visits_silver
# MAGIC union
# MAGIC select 'gold' as layer, count(*) as cnt from vr_fraud_dev.visits_gold

# COMMAND ----------

# MAGIC %sql select 'prod' as layer, count(*) as cnt from vr_fraud_prod.visits_gold

# COMMAND ----------

# MAGIC %sql select visit_id, count(*) as cnt from vr_fraud_dev.visits_gold group by visit_id having cnt > 1 order by cnt desc

# COMMAND ----------

# MAGIC %md ## 02: Data Preparation

# COMMAND ----------

# MAGIC %sql select 'fs'as layer, count(*) as cnt from vr_fraud_dev.fs_atm_visits
