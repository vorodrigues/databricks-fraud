# Databricks notebook source
from databricks_cli.sdk.api_client import ApiClient
db = ApiClient(host='https://e2-demo-field-eng.cloud.databricks.com', token='dapi7e2301c9f82c2681009f7794cf265dcc', api_version='2.1')

# COMMAND ----------

from databricks_cli.sdk.service import JobsService
jobs = JobsService(db)

# COMMAND ----------

import json
with open('./job_settings', 'r') as f:
  settings = json.load(f)['settings']

# COMMAND ----------

import datetime 
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d-%H-%M-%S')

# COMMAND ----------

jobs.submit_run(
  run_name='vr-test-fraud-'+now, 
  existing_cluster_id=settings['existing_cluster_id'],
  notebook_task=settings['notebook_task']
)
