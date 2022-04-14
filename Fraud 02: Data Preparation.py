# Databricks notebook source
# dbutils.widgets.text('db', 'vr_fraud_dev', 'Database')

# COMMAND ----------

db = dbutils.widgets.get('db')
print('DATABASE: '+db)

# COMMAND ----------

# MAGIC %sql USE $db

# COMMAND ----------

# MAGIC %md # Fraud 02: Data Preparation
# MAGIC 
# MAGIC Now that data has been ingested, cleaned and organized, data scientists can explore it to understand patterns and gain insigths.
# MAGIC 
# MAGIC Then they can leverage this knowledge to engineer new features and improve existing ones.
# MAGIC 
# MAGIC And, finally, load them to a Feature Store to share and manage those variables.<br><br>
# MAGIC 
# MAGIC ![](/files/shared_uploads/victor.rodrigues@databricks.com/ml_1.jpg)

# COMMAND ----------

# MAGIC %md ## Visualizing the distribution of fraud

# COMMAND ----------

# MAGIC %sql SELECT COUNT(1), state FROM visits_gold WHERE year = 2016 AND fraud_report = 'Y' GROUP BY state HAVING state IS NOT NULL

# COMMAND ----------

# MAGIC %sql SELECT SUM(amount), month, fraud_report FROM visits_gold WHERE year = 2016 GROUP BY month, fraud_report ORDER BY month

# COMMAND ----------

# MAGIC %md ## Prepare Data

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
# MAGIC   state,
# MAGIC   pos_capability,
# MAGIC   offsite_or_onsite,
# MAGIC   bank,
# MAGIC   checking_savings,
# MAGIC   datediff('2018-01-01', customer_since_date) as customer_lifetime
# MAGIC FROM visits_gold;
# MAGIC 
# MAGIC SELECT * FROM vw_atm_visits LIMIT 100

# COMMAND ----------

# MAGIC %md ## Load Feature Store
# MAGIC 
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a **Delta Lake** table and co-designed with **MLflow**.
# MAGIC 
# MAGIC This will allow **discoverability** and **reusability** of our feature accross our organization, increasing team efficiency.
# MAGIC 
# MAGIC Feature store will bring **traceability** and **governance** in our deployment, knowing which model is dependent of which set of features.<br><br>
# MAGIC 
# MAGIC ![](/files/shared_uploads/victor.rodrigues@databricks.com/fs.jpg)

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md For the first time, use the following code to create a Feature Table:
# MAGIC 
# MAGIC `fs.create_table(
# MAGIC     name=db+".fs_atm_visits",
# MAGIC     df=spark.table("vw_atm_visits"),
# MAGIC     primary_keys=["visit_id"],
# MAGIC     description="ATM Fraud features"
# MAGIC )`

# COMMAND ----------

fs.write_table(
    name=db+".fs_atm_visits",
    df=spark.table("vw_atm_visits"),
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Next Step: AutoML
# MAGIC ## Accelerating Fraud model creation using Databricks Auto-ML
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC 
# MAGIC Databricks simplify model creation and MLOps. However, bootstraping new ML projects can still be long and inefficient. 
# MAGIC 
# MAGIC Instead of creating the same boterplate for each new project, Databricks Auto-ML can automatically generate state of the art models for classification, regression, and forecast.
# MAGIC 
# MAGIC 
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC 
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC 
# MAGIC While this is done using the UI, you can also leverage the [Python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)
