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
            "notebook_path": "/Repos/victor.rodrigues@databricks.com/fraud-dev/Fraud 01: Data Engineering",
            "base_parameters": {
                "db": "vr_fraud_dev",
                "path": "/FileStore/vr/fraud/dev"
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
r = jobs.submit_run(
  run_name='vr-test-fraud-'+now, 
  existing_cluster_id=settings['existing_cluster_id'],
  notebook_task=settings['notebook_task']
)
run_id = r['run_id']
print(run_id)

# COMMAND ----------

from time import sleep
wait = True
while wait:
  r = jobs.get_run_output(run_id=run_id)
  state = r['metadata']['state']['life_cycle_state']
  wait = True if 'result_state' not in r['metadata']['state'] else False
  sleep(5)

# COMMAND ----------

result = r['metadata']['state']['result_state']
print(result)

# COMMAND ----------

if result != 'SUCCESS':
  raise Exception('TEST FAILED!!!')
