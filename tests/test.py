# Databricks notebook source
print('Starting test...')

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

# MAGIC %md # Load Test Settings

# COMMAND ----------

import json
with open('./test_settings', 'r') as f:
  settings = json.load(f)['settings']

# COMMAND ----------

# MAGIC %md # Connect to API

# COMMAND ----------

from databricks_cli.sdk.api_client import ApiClient
db = ApiClient(host=host, token=token, api_version='2.1')

# COMMAND ----------

from databricks_cli.sdk.service import JobsService
jobs = JobsService(db)

# COMMAND ----------

# MAGIC %md # Submit Test

# COMMAND ----------

import datetime 
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d-%H-%M-%S')

print('Submitting run...')
run_id = jobs.submit_run(
  run_name='vr-test-fraud-'+now,
  tasks=settings['tasks']
)['run_id']

r = jobs.get_run(run_id=run_id)
print(r['run_page_url'])

# COMMAND ----------

# MAGIC %md # Evaluate Results

# COMMAND ----------

from time import sleep

while 'result_state' not in r['state']:
  r = jobs.get_run(run_id=run_id)
  sleep(5)

result = r['state']['result_state']
if result == 'SUCCESS':
  print(result)
else:
  raise Exception(result+': '+r['state']['state_message'])
