# Databricks notebook source
dbutils.widgets.text('s3_path', '', 'S3 Path')

# COMMAND ----------

s3_path = dbutils.widgets.get('s3_path')

# COMMAND ----------

from pyspark.sql.functions import when, col

# COMMAND ----------

# MAGIC %md #Catalog

# COMMAND ----------

# MAGIC %sql CREATE CATALOG vr_fraud

# COMMAND ----------

# MAGIC %md #External Location

# COMMAND ----------

# MAGIC %sql CREATE EXTERNAL LOCATION IF NOT EXISTS vr_fraud_location
# MAGIC URL '${s3_path}/fraud'
# MAGIC WITH (STORAGE CREDENTIAL field_demos_credential)

# COMMAND ----------

# MAGIC %md # Dev

# COMMAND ----------

# MAGIC %md ## Database

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS vr_fraud.dev
# MAGIC LOCATION '${s3_path}/fraud/vr_fraud_dev'

# COMMAND ----------

# MAGIC %md ## atm_visits.json

# COMMAND ----------

spark.read.parquet('/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_visits') \
     .withColumn('visit_id', when((col('visit_id') == 42701999) & (col('customer_id') == 204919), 44932650).otherwise(col('visit_id'))) \
     .write.json(f'{s3_path}/fraud/dev/raw/atm_visits')

# COMMAND ----------

# MAGIC %md ## customers_silver

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud.dev.customers_silver 
# MAGIC LOCATION '${s3_path}/fraud/dev/customers_silver' AS
# MAGIC SELECT * FROM parquet.`/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_customers`

# COMMAND ----------

# MAGIC %md ## locations_silver

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud.dev.locations_silver 
# MAGIC LOCATION '${s3_path}/fraud/dev/locations_silver' AS 
# MAGIC SELECT * FROM parquet.`/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_locations`

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
# MAGIC FROM vr_fraud.dev.visits_gold

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists vr_fraud.dev.fs_atm_visits;
# MAGIC drop table if exists vr_fraud.test.fs_atm_visits;
# MAGIC drop table if exists vr_fraud.prod.fs_atm_visits;

# COMMAND ----------

fs.create_table(
    name="vr_fraud.dev.fs_atm_visits",
    df=spark.table("vw_atm_visits").limit(0),
    primary_keys=["visit_id"],
    description="ATM Fraud features [DEV]"
)

# COMMAND ----------

fs.create_table(
    name="vr_fraud.test.fs_atm_visits",
    df=spark.table("vw_atm_visits").limit(0),
    primary_keys=["visit_id"],
    description="ATM Fraud features [TEST]"
)

# COMMAND ----------

fs.create_table(
    name="vr_fraud.prod.fs_atm_visits",
    df=spark.table("vw_atm_visits").limit(0),
    primary_keys=["visit_id"],
    description="ATM Fraud features [PROD]"
)

# COMMAND ----------

# MAGIC %md ## train

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table vr_fraud_dev.tmp as select
# MAGIC   *,
# MAGIC   0.05 * (1 + 0.2 * (rand()-0.5)/0.5) * tx_pos * tx_site * tx_withdrawl * tx_acc * tx_day * tx_hour * tx_amount as tx_final
# MAGIC from (
# MAGIC   select
# MAGIC     *,
# MAGIC     case pos_capability when 'swipe'           then 1 + 0.05 * rand() else 1 end as tx_pos,
# MAGIC     case offsite_or_onsite when 'offsite'      then 1 + 0.05 * rand() else 1 end as tx_site,
# MAGIC     case withdrawl_or_deposit when 'withdrawl' then 1 + 0.05 * rand() else 1 end as tx_withdrawl,
# MAGIC     case checking_savings when 'chk'           then 1 + 0.05 * rand() else 1 end as tx_acc,
# MAGIC     case when day <= 5 or 28 <= day            then 1 + 0.05 * rand() else 1 end as tx_day,
# MAGIC     case when hour <= 6 or hour <= 23          then 1 + 0.05 * rand() else 1 end as tx_hour,
# MAGIC     case when amount > 80                      then 1 + 0.05 * (amount - 80) / 60 else 1 end as tx_amount
# MAGIC   from vr_fraud_dev.fs_atm_visits
# MAGIC   limit 10000
# MAGIC )

# COMMAND ----------

# MAGIC %sql select avg(case when tx_final > 0.0654 then 1 else 0 end) from vr_fraud_dev.tmp

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table vr_fraud_dev.train as select
# MAGIC   visit_id,
# MAGIC   case when tx_final > 0.0654 then 1 else 0 end as fraud_report
# MAGIC from vr_fraud_dev.tmp

# COMMAND ----------

# MAGIC %md - Use AutoML to validate if table is ok

# COMMAND ----------

# MAGIC %md ## test

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_dev.test AS SELECT visit_id FROM vr_fraud_dev.train

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

# MAGIC %md # Test

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

# MAGIC %md # Prod

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
# MAGIC --select 'raw' as layer, count(*) as cnt from JSON.`/FileStore/vr/fraud/dev/raw/atm_visits`
# MAGIC --union
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
