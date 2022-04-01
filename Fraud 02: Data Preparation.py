# Databricks notebook source
# dbutils.widgets.text('db', 'vr_fraud_dev', 'Databse')

# COMMAND ----------

db = dbutils.widgets.get('db')
print('DATABASE: '+db)

# COMMAND ----------

# MAGIC %sql USE $db

# COMMAND ----------

# MAGIC %md # Fraud 02: Data Preparation

# COMMAND ----------

# MAGIC %md ## Visualizing the distribution of fraud

# COMMAND ----------

# MAGIC %sql SELECT COUNT(1), city_state_zip.state state FROM visits_gold WHERE year = 2016 AND fraud_report = 'Y' GROUP BY state HAVING state IS NOT NULL

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
# MAGIC   city_state_zip.state as state,
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

fs.create_table(
    name=db+".fs_atm_visits",
    df=spark.table("vw_atm_visits"),
    primary_keys=["visit_id"],
    description="ATM Fraud features"
)

# COMMAND ----------

fs.write_table(
    name=db+".fs_atm_visits",
    df=spark.table("vw_atm_visits"),
    mode="overwrite"
)
