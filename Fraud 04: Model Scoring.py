# Databricks notebook source
# MAGIC %pip install category_encoders

# COMMAND ----------

# dbutils.widgets.text('db', 'vr_fraud_dev', 'Databse')
# dbutils.widgets.text('path', 'vr_fraud_dev', 'Path')

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

# COMMAND ----------

# MAGIC %md ## Step 1: Score Model

# COMMAND ----------

# DBTITLE 1,Create Feature Store Client
from databricks import feature_store
import os, shutil
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Generate Predictions with Production Model
model_name = 'VR Fraud Analytics'

preds = fs.score_batch(
    'models:/%s/Production' % model_name,
    spark.table(db+'.test')
)['visit_id','prediction']

display(preds)

# COMMAND ----------

# MAGIC %md ## Step 2: Save Predictions
# MAGIC 
# MAGIC Regardless of whether our intent is to use the [Microsoft Dynamics CRM Common Data Service](https://docs.microsoft.com/en-us/powerapps/developer/common-data-service/import-data) or [Salesforce DataLoader](https://developer.salesforce.com/docs/atlas.en-us.dataLoader.meta/dataLoader/data_loader.htm), we need to produce a UTF-8, delimited text file with a header row. We can deliver such a file as follows:

# COMMAND ----------

# MAGIC %md ### Step 2a: Delta Table

# COMMAND ----------

# DBTITLE 1,Save Predictions to a Delta Table
preds.writeTo('preds').createOrReplace()

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
