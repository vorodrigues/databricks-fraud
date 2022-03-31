# Databricks notebook source
dbutils.widgets.text('table', 'atm_customers', 'Table')

# COMMAND ----------

table = dbutils.widgets.get('table')

# COMMAND ----------

spark.read.parquet("/mnt/databricks-datasets-private/ML/fighting_atm_fraud/" + table)
