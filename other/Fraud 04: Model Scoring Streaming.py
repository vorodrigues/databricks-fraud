# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC This is an auto-generated notebook to perform Streaming inference in a Delta Live Tables pipeline using a selected model from the model registry. This feature is in preview.
# MAGIC
# MAGIC ## Instructions:
# MAGIC Please [add](https://docs.databricks.com/workflows/delta-live-tables/delta-live-tables-ui.html#edit-settings) this notebook to your Delta Live Tables pipeline as an additional notebook library,
# MAGIC or [create](https://docs.databricks.com/workflows/delta-live-tables/delta-live-tables-ui.html#create-a-pipeline) a new pipeline with this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input Table Name and schema
# MAGIC The cell below retrieves the Delta Live Tables table name and table schema from the given Delta table.

# COMMAND ----------

from delta.tables import *

input_delta_table = DeltaTable.forName(spark, "vr_fraud_dev.fs_atm_visits")

# The Delta Live Tables table name for the input table that will be used in the pipeline code below.
input_dlt_table_name = "fs_atm_visits"

# The input table schema stored as an array of strings. This is used to pass in the schema to the model predict udf.
input_dlt_table_columns = input_delta_table.toDF().columns

# COMMAND ----------

# MAGIC %md ## Load model as UDF and restore environment
# MAGIC **Note**: If the model does not return double values, override `result_type` to the desired type.

# COMMAND ----------

# MAGIC %md ### FS

# COMMAND ----------

import mlflow

model_uri = f"models:/VR Fraud RT Model/Production"

# create spark user-defined function for model prediction.
# Note: : Here we use virtualenv to restore the python environment that was used to train the model.
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager='virtualenv')

# COMMAND ----------

# MAGIC %md ### Regular

# COMMAND ----------

model_uri = "models:/VR Fraud RT Model/production"

# create spark user-defined function for model prediction.
# Note: : Here we use virtualenv to restore the python environment that was used to train the model.
predict_no_fs = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager='virtualenv')

# COMMAND ----------

# MAGIC %md ## Run inference - FS + withColumn

# COMMAND ----------

columns = spark.table('vr_fraud_dev.fs_atm_visits').columns
columns.remove('visit_id')
print(columns)

# COMMAND ----------

from pyspark.sql.functions import struct

spark.readStream.table('vr_fraud_dev.fs_atm_visits').limit(1000) \
  .withColumn('prediction', predict(struct(columns))) \
  .writeStream \
  .option('checkpointLocation', '/tmp/vr_fraud_dev.fs_atm_visits') \
  .toTable('vr_fraud_dev.preds_stream')

# COMMAND ----------

# MAGIC %md ## Run inference - FS + foreachbatch

# COMMAND ----------

# MAGIC %md ## Run inference - No + withColumn

# COMMAND ----------

columns = spark.table('vr_fraud_dev.fs_atm_visits').columns
columns.remove('visit_id')
print(columns)

# COMMAND ----------

from pyspark.sql.functions import struct

spark.readStream.table('vr_fraud_dev.fs_atm_visits').limit(1000) \
  .withColumn('prediction', predict_no_fs(struct(columns))) \
  .writeStream \
  .option('checkpointLocation', '/tmp/vr_fraud_dev.fs_atm_visits') \
  .toTable('vr_fraud_dev.preds_stream')

# COMMAND ----------

# MAGIC %md ## Clear checkpoint

# COMMAND ----------

dbutils.fs.rm('/tmp/vr_fraud_dev.fs_atm_visits', recurse=True)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Sanity check

# COMMAND ----------

import mlflow

# COMMAND ----------

model_uri = "models:/VR Fraud RT Model/1"

# create spark user-defined function for model prediction.
# Note: : Here we use virtualenv to restore the python environment that was used to train the model.
predict_no_fs = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager='virtualenv')

# COMMAND ----------

columns = spark.table('vr_fraud_dev.fs_atm_visits').columns
columns.remove('visit_id')
print(columns)

# COMMAND ----------

columns = ['amount', 'bank', 'checking_savings', 'customer_lifetime', 'day', 'hour', 'min', 'month', 'offsite_or_onsite', 'pos_capability', 'state', 'withdrawl_or_deposit', 'year']
print(columns)

# COMMAND ----------

from pyspark.sql.functions import struct

display(
  spark.read.table('vr_fraud_dev.fs_atm_visits').limit(1000)
    .select(columns)
    .withColumn('prediction', predict_no_fs())
)
