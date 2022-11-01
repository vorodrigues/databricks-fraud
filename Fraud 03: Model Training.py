# Databricks notebook source
# MAGIC %pip install category_encoders

# COMMAND ----------

# dbutils.widgets.text('db', 'vr_fraud_dev', 'Database')

# COMMAND ----------

db = dbutils.widgets.get('db')
print('DATABASE: '+db)

# COMMAND ----------

# MAGIC %sql USE DATABASE $db

# COMMAND ----------

# MAGIC %md # Fraud 03: Model Training
# MAGIC 
# MAGIC The next step is to train lots of different models using different algorithms and parameters in search for the one that optimally solves our business problem.
# MAGIC 
# MAGIC That's where the **Spark** + **HyperOpt** + **MLflow** framework can be leveraged to easily distribute the training proccess across a cluster, efficiently optimize hyperparameters and track all experiments in order to quickly evaluate many models, choose the best one and guarantee its reproducibility.<br><br>
# MAGIC 
# MAGIC ![](/files/shared_uploads/victor.rodrigues@databricks.com/ml_2.jpg)

# COMMAND ----------

# MAGIC %md ## Load data from Feature Store

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import FeatureLookup

fs = feature_store.FeatureStoreClient()

feature_lookups = [
    FeatureLookup(
      table_name = db+'.fs_atm_visits',
      feature_names = None,
      lookup_key = ['visit_id']
    )
]

training_set = fs.create_training_set(
  df = spark.table(db+'.train'),
  feature_lookups = feature_lookups,
  label = 'fraud_report',
  exclude_columns = ['visit_id']
)

df = training_set.load_df()
display(df)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# split train and test datasets
train, test = train_test_split(df.toPandas(), train_size=0.7)

# separate features and labels
train = train
X_train_raw = train.drop('fraud_report', axis=1)
y_train = train['fraud_report']

# separate features and labels
test = test
X_test_raw = test.drop('fraud_report', axis=1)
y_test = test['fraud_report']

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from category_encoders.woe import WOEEncoder
set_config(display="diagram")

# Separate variables
numVars = ['day', 'hour','amount','customer_lifetime']
catVars = ['bank', 'checking_savings', 'offsite_or_onsite', 'pos_capability', 'state', 'withdrawl_or_deposit']

# Handling numerical data
num = Pipeline(steps=[
    ('std', StandardScaler()),
    ('imp', SimpleImputer(strategy='mean'))
])

# Handling categorical data
cat = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', WOEEncoder())
])

# Preprocessor
pre = Pipeline(steps=[(
  'preprocessor', ColumnTransformer(transformers=[
    ('num', num, numVars),
    ('cat', cat, catVars)
  ])
)])

# Transform data
X_train = pre.fit_transform(X_train_raw, y_train)
X_test = pre.transform(X_test_raw)

display(pre)

# COMMAND ----------

# MAGIC %md ## Tune XGBClassifier

# COMMAND ----------

# MAGIC %md ### Define Experiment
# MAGIC 
# MAGIC The XGBClassifier makes available a [wide variety of hyperparameters](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) which can be used to tune model training.  Using some knowledge of our data and the algorithm, we might attempt to manually set some of the hyperparameters. But given the complexity of the interactions between them, it can be difficult to know exactly which combination of values will provide us the best model results.  It's in scenarios such as these that we might perform a series of model runs with different hyperparameter settings to observe how the model responds and arrive at an optimal combination of values.
# MAGIC 
# MAGIC Using hyperopt, we can automate this task, providing the hyperopt framework with a range of potential values to explore.  Calling a function which trains the model and returns an evaluation metric, hyperopt can through the available search space to towards an optimum combination of values.
# MAGIC 
# MAGIC For model evaluation, we will be using the Area Under the Curve (AUC) score which increases towards 1.0 as the model improves.  Because hyperopt recognizes improvements as our evaluation metric declines, we will use `-1 * AUC` as our loss metric within the framework. 
# MAGIC 
# MAGIC Putting this all together, we might arrive at model training and evaluation function as follows:

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def evaluate_model(hyperopt_params):
  
  # accesss replicated input data
  X_train_input = X_train_broadcast.value
  y_train_input = y_train_broadcast.value
  X_test_input = X_test_broadcast.value
  y_test_input = y_test_broadcast.value  
  
  # configure model parameters
  params = hyperopt_params
  
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train_input, y_train_input)
  
  # predict
  prob_train = model.predict_proba(X_train_input)
  prob_test = model.predict_proba(X_test_input)
  
  # evaluate
  auc_train = roc_auc_score(y_train_input, prob_train[:,1])
  auc_test = roc_auc_score(y_test_input, prob_test[:,1])
  
  # log model
  mlflow.log_metric('train_auc', auc_train)
  mlflow.log_metric('test_auc', auc_test)
  
  # invert metric for hyperopt
  loss = -1 * auc_test  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md The first part of the model evaluation function retrieves from memory replicated copies of our training and testing feature and label sets.  Our intent is to leverage SparkTrials in combination with hyperopt to parallelize the training of models across a Spark cluster, allowing us to perform multiple, simultaneous model training evaluation runs and reduce the overall time required to navigate the seach space.  By replicating our datasets to the worker nodes of the cluster, a task performed in the next cell, copies of the data needed for training and evaluation can be efficiently made available to the function with minimal networking overhead:
# MAGIC 
# MAGIC **NOTE** See the Distributed Hyperopt [best practices documentation](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html#handle-datasets-of-different-orders-of-magnitude-notebook) for more options for data distribution.

# COMMAND ----------

X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)

# COMMAND ----------

# MAGIC %md ### Define Search Space
# MAGIC The hyperparameter values delivered to the function by hyperopt are derived from a search space defined in the next cell.  Each hyperparameter in the search space is defined using an item in a dictionary, the name of which identifies the hyperparameter and the value of which defines a range of potential values for that parameter.  When defined using *hp.choice*, a parameter is selected from a predefined list of values.  When defined *hp.loguniform*, values are generated from a continuous range of values.  When defined using *hp.quniform*, values are generated from a continuous range but truncated to a level of precision identified by the third argument  in the range definition.  Hyperparameter search spaces in hyperopt may be defined in many other ways as indicated by the library's [online documentation](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):  

# COMMAND ----------

from sklearn.utils.class_weight import compute_class_weight
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
import numpy as np

# define minimum positive class scale factor (as shown in previous notebook)
weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train), 
  y=y_train
)
scale = weights[1]/weights[0]

# define hyperopt search space
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 10, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
    ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.40))     # learning rate for XGBoost
    ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1)                   # minimum number of instances per node
    ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
    ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
    ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
    ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
    ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(scale), np.log(scale*1.1))   # weight to assign positive label to manage imbalance
}

# COMMAND ----------

# MAGIC %md ### Run Experiment
# MAGIC 
# MAGIC The remainder of the model evaluation function is fairly straightforward.  We simply train and evaluate our model and return our loss value, *i.e.* `-1 * AUC`, as part of a dictionary interpretable by hyperopt.  Based on returned values, hyperopt will generate a new set of hyperparameter values from within the search space definition with which it will attempt to improve our metric. We will limit the number of hyperopt evaluations to 250 simply based on a few trail runs we performed (not shown).  The larger the potential search space and the degree to which the model (in combination with the training dataset) responds to different hyperparameter combinations determines how many iterations are required for hyperopt to arrive at locally optimal values.  You can examine the output of the hyperopt run to see how our loss metric slowly improves over the course of each of these evaluations:

# COMMAND ----------

import mlflow

# perform evaluation
with mlflow.start_run(run_name='XGBClassifer'):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=SparkTrials(parallelism=8), # set to the number of available cores
    verbose=True
  )

# COMMAND ----------

# MAGIC %md ### Save Best Model

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# COMMAND ----------

from mlflow.models.signature import infer_signature

# train model with optimal settings 
with mlflow.start_run(run_name='XGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
   
  # configure params
  params = space_eval(search_space, argmin)
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])
  if 'scale_pos_weight' in params: params['scale_pos_weight']=int(params['scale_pos_weight'])    
  params['tree_method']='hist'
  params['predictor']='cpu_predictor'
  mlflow.log_params(params)
  
  # train
  model = Pipeline(steps=[
    ('pre', pre),
    ('clf', XGBClassifier(**params))
  ])
  model.fit(X_train_raw, y_train)
  
  # predict
  prob_train = model.predict_proba(X_train_raw)
  prob_test = model.predict_proba(X_test_raw)
  
  # evaluate
  auc_train = roc_auc_score(y_train, prob_train[:,1])
  auc_test = roc_auc_score(y_test, prob_test[:,1])

  # log model
  wrappedModel = SklearnModelWrapper(model)
  signature = infer_signature(X_train_raw, prob_train)
  mlflow.pyfunc.log_model(python_model=wrappedModel, artifact_path='model', signature=signature)
  mlflow.log_metric('train_auc', auc_train)
  mlflow.log_metric('test_auc', auc_test)

  print('Model logged under run_id "{0}" with AUC score of {1:.5f}'.format(run_id, auc_test))
  display(model)

# COMMAND ----------

# MAGIC %md ## Register Champion Model
# MAGIC 
# MAGIC After choosing a model that best fits our needs, we can then go ahead and kick off its operationalization proccess.
# MAGIC 
# MAGIC The first step is to register it to the **Model Registry**, where we can version, manage its life cycle with an workflow and track/audit all changes.<br><br>
# MAGIC 
# MAGIC ![](/files/shared_uploads/victor.rodrigues@databricks.com/ml_3.jpg)

# COMMAND ----------

model_name = 'VR Fraud Model'

# COMMAND ----------

fs.log_model(
  wrappedModel,
  "model",
  flavor=mlflow.pyfunc,
  training_set=training_set,
  registered_model_name=model_name
)
