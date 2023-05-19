# Databricks notebook source
import mlflow

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

# DBTITLE 1,Intended for demo purposes only
dbutils.fs.rm('/tmp/vr_predict_ft', recurse=True)
spark.sql('drop table vr_fraud_dev.preds_stream_ft')
dbutils.fs.rm('/tmp/vr_predict', recurse=True)
spark.sql('drop table vr_fraud_dev.preds_stream')

# COMMAND ----------

# MAGIC %md # Fraud 04: Model Scoring - Streaming
# MAGIC
# MAGIC Finally, the model can be scored on a new dataset to get predictions.
# MAGIC
# MAGIC Models can be deployed as a batch, stream or real time proccess.<br><br>
# MAGIC
# MAGIC ![](/files/shared_uploads/victor.rodrigues@databricks.com/ml_4.jpg)

# COMMAND ----------

# MAGIC %md ### Model registered with Feature Store

# COMMAND ----------

# DBTITLE 1,Load Model as a Spark UDF
model_uri = f"models:/VR Fraud ST Model/Production"

run_id = mlflow.models.get_model_info(model_uri).run_id
predict_ft = mlflow.pyfunc.spark_udf(spark, f'runs:/{run_id}/model/data/feature_store/raw_model', result_type="double", env_manager='local')

# COMMAND ----------

# DBTITLE 1,Score Model
spark.readStream.table('vr_fraud_dev.fs_atm_visits') \
  .withColumn('prediction', predict_ft()) \
  .writeStream \
  .option('checkpointLocation', '/tmp/vr_predict_ft') \
  .toTable('vr_fraud_dev.preds_stream_ft')

# COMMAND ----------

# DBTITLE 1,Visualize Predictions
display(spark.table('vr_fraud_dev.preds_stream_ft'))

# COMMAND ----------

# MAGIC %md ### Model registered without Feature Store

# COMMAND ----------

# DBTITLE 1,Load Model as a Spark UDF
model_uri = f"models:/VR Fraud ST Model/Staging"

# create spark user-defined function for model prediction.
# Note: : Here we use virtualenv to restore the python environment that was used to train the model.
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager='virtualenv')

# COMMAND ----------

# DBTITLE 1,Score Model
spark.readStream.table('vr_fraud_dev.fs_atm_visits') \
  .withColumn('prediction', predict()) \
  .writeStream \
  .option('checkpointLocation', '/tmp/vr_predict') \
  .toTable('vr_fraud_dev.preds_stream')

# COMMAND ----------

# DBTITLE 1,Visualize Predictions
display(spark.table('vr_fraud_dev.preds_stream'))
