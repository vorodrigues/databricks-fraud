# Databricks notebook source
print('Starting deployment...')

# COMMAND ----------

# MAGIC %md # Parse Arguments

# COMMAND ----------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--host")
parser.add_argument("--token")
args = parser.parse_args()

host = args.host
token = args.token

# COMMAND ----------

host = 'https://e2-demo-west.cloud.databricks.com'
token = 'dapib5589f9d0964665829aee60fa85aae5b'

# COMMAND ----------

# MAGIC %md # Connect to API

# COMMAND ----------

from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.sdk.service import ReposService
db = ApiClient(host=host, token=token, api_version='2.0')
repos = ReposService(db)

# COMMAND ----------

from databricks_cli.sdk.service import JobsService
db = ApiClient(host=host, token=token, api_version='2.1')
jobs = JobsService(db)

# COMMAND ----------

# MAGIC %md # Update Repos

# COMMAND ----------

print('Updating repos...')
repos_id = repos.list_repos(path_prefix='/Repos/victor.rodrigues@databricks.com/fraud-prod')['repos'][0]['id']
repos.update_repo(id=repos_id, branch='prod')

# COMMAND ----------

# MAGIC %md # Update Job

# COMMAND ----------

import json
with open('./job_settings', 'r') as f:
  settings = json.load(f)['settings']

# COMMAND ----------

from os.path import exists

# Update existing job
if exists('./job_id'):
  print('Updating job...')
  with open('./job_id', 'r') as f:
    job_id = f.read()
  jobs.reset_job(job_id=job_id, new_settings=settings)

# Create new job
else:
  print('Creating new job...')
  job_id = jobs.create_job(name=settings['name'], tasks=settings['tasks'])['job_id']
  with open('./job_id', 'w') as f:
    f.write(str(job_id))
  print(job_id)
