# Databricks notebook source
from databricks_cli.sdk.api_client import ApiClient
db = ApiClient(host='https://demo.cloud.databricks.com', token='dapi2331c6e00419d5466bad7d8bc58e2678', api_version='2.1')

# COMMAND ----------

from databricks_cli.sdk.service import JobsService
jobs = JobsService(db)

# COMMAND ----------

import json

f = {
    "settings": {
        "existing_cluster_id": "0331-230110-ojy94y49",
        "notebook_task": {
            "notebook_path": "/Repos/victor.rodrigues@databricks.com/fraud-dev/test-nb",
            "base_parameters": {
                "table": "atm_customers"
            }
        },
        "timeout_seconds": 0,
        "email_notifications": {},
        "name": "test-nb",
        "max_concurrent_runs": 1
    }
}

settings = json.loads(json.dumps(f))['settings']

# COMMAND ----------

import datetime 
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d-%H-%M-%S')

# COMMAND ----------

print('Submitting run...')
rid = jobs.submit_run(
  run_name='vr-test-fraud-'+now, 
  existing_cluster_id=settings['existing_cluster_id'],
  notebook_task=settings['notebook_task']
)
print(rid)
