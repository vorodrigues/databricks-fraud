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

# MAGIC %md # Query existing jobs

# COMMAND ----------

import json
with open('./job_settings', 'r') as f:
  settings = json.load(f)['settings']

# COMMAND ----------

name = settings['name']

job_list = db.perform_query(method="get", version="2.1", path="/jobs/list", data={"name": name})['jobs']

# COMMAND ----------

# MAGIC %md # Create or Update Job

# COMMAND ----------

# Check for duplicated jobs
if len(job_list) > 1:
  raise Exception(
    f"""There are more than one jobs with name {name}.
        Please delete duplicated jobs first."""
  )

# Create new job
elif not job_list:
  print('Creating new job...')
  job_id = jobs.create_job(name=name, tasks=settings['tasks'])['job_id']
  print(f'job_id = {job_id}')

# Update existing job
else:
  job_id = job_list[0]['job_id']
  print(f'Updating job {name}...')
  jobs.reset_job(job_id=job_id, new_settings=settings)
