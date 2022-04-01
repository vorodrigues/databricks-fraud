# Databricks notebook source
# dbutils.widgets.text('host', 'https://demo.cloud.databricks.com')
# dbutils.widgets.text('token', '')

# COMMAND ----------

dbutils.widgets.get('host')
dbutils.widgets.get('token')

# COMMAND ----------

print('Starting test...')

# COMMAND ----------

from databricks_cli.sdk.api_client import ApiClient
db = ApiClient(host=host, token=token, api_version='2.1')

# COMMAND ----------

from databricks_cli.sdk.service import JobsService
jobs = JobsService(db)

# COMMAND ----------

import json

f = {
    "settings": {
        "timeout_seconds": 0,
        "email_notifications": {},
        "name": "VR Fraud",
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "existing_cluster_id": "0331-230110-ojy94y49",
                "notebook_task": {
                    "notebook_path": "/Repos/victor.rodrigues@databricks.com/fraud-dev/Fraud 01: Data Engineering",
                    "base_parameters": {
                        "db": "vr_fraud_test",
                        "path": "/FileStore/vr/fraud/test"
                    }
                },
                "timeout_seconds": 0,
                "email_notifications": {},
                "task_key": "data-engineering",
                "description": ""
            },
            {
                "existing_cluster_id": "0331-230110-ojy94y49",
                "notebook_task": {
                    "notebook_path": "/Repos/victor.rodrigues@databricks.com/fraud-dev/Fraud 02: Data Preparation",
                    "base_parameters": {
                        "db": "vr_fraud_test"
                    }
                },
                "timeout_seconds": 0,
                "email_notifications": {},
                "task_key": "data-preparation",
                "depends_on": [
                    {
                        "task_key": "data-engineering"
                    }
                ]
            },
            {
                "existing_cluster_id": "0331-230110-ojy94y49",
                "notebook_task": {
                    "notebook_path": "/Repos/victor.rodrigues@databricks.com/fraud-dev/Fraud 04: Model Scoring",
                    "base_parameters": {
                        "db": "vr_fraud_test",
                        "path": "/FileStore/vr/fraud/test"
                    }
                },
                "timeout_seconds": 0,
                "email_notifications": {},
                "task_key": "model-scoring",
                "depends_on": [
                    {
                        "task_key": "data-preparation"
                    }
                ]
            }
        ]
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
  tasks=settings['tasks']
)
run_id = r['run_id']

# COMMAND ----------

from time import sleep

r = jobs.get_run(run_id=run_id)
print(r['run_page_url'])

wait = True if 'result_state' not in r['state'] else False

wait = True
while wait:
  r = jobs.get_run(run_id=run_id)
  wait = True if 'result_state' not in r['state'] else False
  sleep(5)

# COMMAND ----------

result = r['state']['result_state']
if result == 'SUCCESS':
  print(result)
else:
  raise Exception(result+': '+r['state']['state_message'])
