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

import mlflow

model_uri = f"models:/VR Fraud Model/Production"

# create spark user-defined function for model prediction.
# Note: : Here we use virtualenv to restore the python environment that was used to train the model.
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager='virtualenv')

# COMMAND ----------

# MAGIC %md ## Run inference & Write predictions to output
# MAGIC **Note**: If you want to rename the output Delta table in the pipeline later on, please rename the function below: `fraud_predictions()`.

# COMMAND ----------

import dlt
from pyspark.sql.functions import struct

@dlt.table(
  comment="DLT for predictions scored by VR Fraud Model model based on vr_fraud_dev.fs_atm_visits Delta table.",
  table_properties={
    "quality": "gold"
  }
)
def fraud_predictions():
  return (
    dlt.read(input_dlt_table_name)
    .withColumn('prediction', predict(struct(input_dlt_table_columns)))
  )

