# Databricks notebook source
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("host")
parser.add_argument("token")
args = parser.parse_args()

host = args.host
token = args.token

# COMMAND ----------

print('Starting deployment...')

# COMMAND ----------

from databricks_cli.sdk.api_client import ApiClient
db = ApiClient(host=host, token=token, api_version='2.0')

# COMMAND ----------

from databricks_cli.sdk.service import ReposService
repos = ReposService(db)

# COMMAND ----------

print('Updating repos...')
repos.update_repo(id=12511536, branch='prod')

# COMMAND ----------

from databricks_cli.sdk.service import JobsService
db = ApiClient(host=host, token=token, api_version='2.1')
jobs = JobsService(db)

# COMMAND ----------

import json

f = {
    "settings": {
        "timeout_seconds": 0,
        "email_notifications": {},
        "name": "VR Fraud Analytics",
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "existing_cluster_id": "0331-230110-ojy94y49",
                "notebook_task": {
                    "notebook_path": "/Repos/victor.rodrigues@databricks.com/fraud-dev/Fraud 01: Data Engineering",
                    "base_parameters": {
                        "db": "vr_fraud_prod",
                        "path": "/FileStore/vr/fraud/prod"
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
                        "db": "vr_fraud_prod"
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
                        "db": "vr_fraud_prod",
                        "path": "/FileStore/vr/fraud/prod"
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

from os.path import exists

# Update existing job
if exists('./job.id'):
  print('Updating job...')
  with open('./job.id', 'r') as f:
    job_id = f.read()
  jobs.reset_job(job_id=job_id, new_settings=settings)

# Create new job
else:
  print('Creating new job...')
  job_id = jobs.create_job(name=settings['name'], tasks=settings['tasks'])['job_id']
  with open('./job.id', 'w') as f:
    f.write(str(job_id))
  print(job_id)
