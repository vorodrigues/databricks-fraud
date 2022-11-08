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
# MAGIC 
# MAGIC Let's analyze our dataset to understand fraud behaviour in general.
# MAGIC 
# MAGIC We can query the data using our preferred language and then create visualizations without wiritting a line of code.

# COMMAND ----------

# MAGIC %sql SELECT SUM(amount), hour, fraud_report FROM visits_gold GROUP BY hour, fraud_report ORDER BY hour

# COMMAND ----------

# MAGIC %sql SELECT COUNT(1) as visits, state FROM visits_gold WHERE year = 2016 AND fraud_report = 'Y' GROUP BY state HAVING state IS NOT NULL

# COMMAND ----------

# MAGIC %md ## Prepare Data

# COMMAND ----------

# MAGIC %md ### BambooLib
# MAGIC 
# MAGIC Databricks alllows you to quickly create your pipelines using a **no-code UI**. You can interact with your data and simply select transformations you want to apply directly on the notebook. BambooLib will automatically generate the code for you in a **glass-box approach**: you can review and modify the code as much as you want.
# MAGIC 
# MAGIC In this notebook we're going to:<br><br>
# MAGIC 
# MAGIC - Load transaction (visit), customer and location data
# MAGIC - Filter desired visits
# MAGIC - Drop rows with missing values
# MAGIC - Drop duplicates
# MAGIC - Combine all datasets
# MAGIC - Convert date columns
# MAGIC - Create customer_lifetime variable

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

import bamboolib as bam
bam

# COMMAND ----------

# MAGIC %md ### Pandas
# MAGIC 
# MAGIC Databricks ML Runtime includes all major Data Science tools and libraries by default, so you don't need to worry about setting up your environment.
# MAGIC 
# MAGIC Let's use Pandas to prepare our data for training ML models.

# COMMAND ----------

import pandas as pd

visits_df = pd.read_parquet('/dbfs/FileStore/vr/fraud/dev/parquet/visits_silver')
customers_df = pd.read_parquet('/dbfs/FileStore/vr/fraud/dev/parquet/customers_silver')
locations_df = pd.read_parquet('/dbfs/FileStore/vr/fraud/dev/parquet/locations_silver')

# Step: Keep rows where amount > 0
visits_df = visits_df.loc[visits_df['amount'] > 0]

# Step: Drop missing values in [All columns]
visits_df = visits_df.dropna()

# Step: Drop duplicates based on ['amount', 'atm_id', 'customer_id', 'day', 'fraud_report', 'hour', 'min', 'month', 'sec', 'visit_id', 'withdrawl_or_deposit', 'year', 'date_visit']
visits_df = visits_df.drop_duplicates(keep='first')

# Step: Inner Join with customer_df where customer_id=customer_id
visits_df = pd.merge(visits_df, customers_df, how='inner', on=['customer_id'])

# Step: Inner Join with locations_df where atm_id=atm_id
visits_df = pd.merge(visits_df, locations_df, how='inner', on=['atm_id'])

# Step: Change data type of customer_since_date to Datetime
visits_df['customer_since_date'] = pd.to_datetime(visits_df['customer_since_date'], format='%Y-%m-%d')

# Step: Change data type of date_visit to Datetime
visits_df['date_visit'] = pd.to_datetime(visits_df['date_visit'], format='%Y-%m-%d')

# Step: Create new column 'customer_lifetime' from formula 'date_visit - customer_since_date'
visits_df['customer_lifetime'] = visits_df['date_visit'] - visits_df['customer_since_date']

visits_df

# COMMAND ----------

# MAGIC %md-sandbox ### Pandas on Spark
# MAGIC 
# MAGIC <div style="float:right ;">
# MAGIC   <img src="https://raw.githubusercontent.com/databricks/koalas/master/Koalas-logo.png" width="150"/>
# MAGIC </div>
# MAGIC 
# MAGIC Using Databricks, data scientists don't have to learn a new API to analyse data and deploy new model in production
# MAGIC 
# MAGIC * If you model is small and fit in a single node, you can use a Single Node cluster with pandas directly
# MAGIC * If your data grow, no need to re-write the code. Just switch to pandas on spark and your cluster will parallelize your compute out of the box.
# MAGIC 
# MAGIC #### Scale Pandas with Databricks Runtime as backend
# MAGIC 
# MAGIC 
# MAGIC One of the known limitations in pandas is that it does not scale with your data volume linearly due to single-machine processing. For example, pandas fails with out-of-memory if it attempts to read a dataset that is larger than the memory available in a single machine.
# MAGIC 
# MAGIC 
# MAGIC Pandas API on Spark overcomes the limitation, enabling users to work with large datasets by leveraging Spark!
# MAGIC 
# MAGIC 
# MAGIC **As result, Data Scientists can access dataset in Unity Catalog with simple SQL or spark command, and then switch to the API they know (pandas) best without having to worry about the table size and scalability!**
# MAGIC 
# MAGIC *Note: Starting with spark 3.2, pandas API are directly part of spark runtime, no need to import external library!*

# COMMAND ----------

import pyspark.pandas as ps

visits_df = spark.table(f"{db}.visits_silver").pandas_api()
customers_df = spark.table(f"{db}.customers_silver").pandas_api()
locations_df = spark.table(f"{db}.locations_silver").pandas_api()

# Step: Keep rows where amount > 0
visits_df = visits_df.loc[visits_df['amount'] > 0]

# Step: Drop missing values in [All columns]
visits_df = visits_df.dropna()

# Step: Drop duplicates based on ['amount', 'atm_id', 'customer_id', 'day', 'fraud_report', 'hour', 'min', 'month', 'sec', 'visit_id', 'withdrawl_or_deposit', 'year', 'date_visit']
visits_df = visits_df.drop_duplicates(keep='first')

# Step: Inner Join with customer_df where customer_id=customer_id
visits_df = ps.merge(visits_df, customers_df, how='inner', on=['customer_id'])

# Step: Inner Join with locations_df where atm_id=atm_id
visits_df = ps.merge(visits_df, locations_df, how='inner', on=['atm_id'])

# Step: Change data type of customer_since_date to Datetime
visits_df['customer_since_date'] = ps.to_datetime(visits_df['customer_since_date'], format='%Y-%m-%d')

# Step: Change data type of date_visit to Datetime
visits_df['date_visit'] = ps.to_datetime(visits_df['date_visit'], format='%Y-%m-%d')

# Step: Create new column 'customer_lifetime' from formula 'date_visit - customer_since_date'
visits_df['customer_lifetime'] = visits_df['date_visit'] - visits_df['customer_since_date']

display(visits_df)

# COMMAND ----------

# MAGIC %md ### Pandas + SQL
# MAGIC 
# MAGIC Pandas on Spark dataframes can also be queried using plain SQL, allowing a perfect match between Pandas Python API and SQL usage

# COMMAND ----------

visits_df = ps.sql('''
  SELECT
    visit_id,
    year,
    month,
    day,
    hour,
    min,
    amount,
    withdrawl_or_deposit,
    city_state_zip.state as state,
    pos_capability,
    offsite_or_onsite,
    bank,
    checking_savings,
    datediff(date_visit, customer_since_date) as customer_lifetime
  FROM {visits_df}
''', visits_df = visits_df)

display(visits_df)

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
# MAGIC     df=visits_df.to_spark(),
# MAGIC     primary_keys=["visit_id"],
# MAGIC     description="ATM Fraud features"
# MAGIC )`

# COMMAND ----------

fs.write_table(
    name=db+".fs_atm_visits",
    df=visits_df.to_spark(),
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
