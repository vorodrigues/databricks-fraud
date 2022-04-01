# Databricks notebook source
# MAGIC %md # Dev Tables

# COMMAND ----------

# MAGIC %md ## Database

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS vr_fraud_dev;

# COMMAND ----------

# MAGIC %md ## atm_visits.json

# COMMAND ----------

spark.read.parquet('/mnt/databricks-datasets-private/ML/fighting_atm_fraud/atm_visits') \
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

# MAGIC %md ## TODO: train

# COMMAND ----------

# MAGIC %md ## TODO: test

# COMMAND ----------

# MAGIC %md # Test Tables

# COMMAND ----------

# MAGIC %md ## Database

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS vr_fraud_test;

# COMMAND ----------

# MAGIC %md ## atm_visits.json

# COMMAND ----------

spark.read.json('/FileStore/vr/fraud/raw/atm_visits') \
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

# MAGIC %md ## Feature Store

# COMMAND ----------

# MAGIC %md Use notebooks 1 and 2 to create the feature table

# COMMAND ----------

# MAGIC %md ## test

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE vr_fraud_test.test AS SELECT visit_id FROM vr_fraud_test.visits_gold

# COMMAND ----------

# MAGIC %md # Prod Tables

# COMMAND ----------


