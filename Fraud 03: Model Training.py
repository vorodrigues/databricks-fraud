# Databricks notebook source
# MAGIC %pip install category_encoders

# COMMAND ----------

db = dbutils.widgets.get('db')

# COMMAND ----------

# MAGIC %sql USE DATABASE $db

# COMMAND ----------

# MAGIC %md # Fraud 03: Model Training

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
from category_encoders.woe import WOEEncoder

# Separate variables
numVars = ['hour','amount','customer_lifetime']
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

# COMMAND ----------

from sklearn import set_config
set_config(display="diagram")
pre

# COMMAND ----------

# MAGIC %md ## Tune XGBClassifier

# COMMAND ----------

# MAGIC %md ### Define Experiment
# MAGIC 
# MAGIC The XGBClassifier makes available a [wide variety of hyperparameters](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) which can be used to tune model training.  Using some knowledge of our data and the algorithm, we might attempt to manually set some of the hyperparameters. But given the complexity of the interactions between them, it can be difficult to know exactly which combination of values will provide us the best model results.  It's in scenarios such as these that we might perform a series of model runs with different hyperparameter settings to observe how the model responds and arrive at an optimal combination of values.
# MAGIC 
# MAGIC Using hyperopt, we can automate this task, providing the hyperopt framework with a range of potential values to explore.  Calling a function which trains the model and returns an evaluation metric, hyperopt can through the available search space to towards an optimum combination of values.
# MAGIC 
# MAGIC For model evaluation, we will be using the average precision (AP) score which increases towards 1.0 as the model improves.  Because hyperopt recognizes improvements as our evaluation metric declines, we will use -1 * the AP score as our loss metric within the framework. 
# MAGIC 
# MAGIC Putting this all together, we might arrive at model training and evaluation function as follows:

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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
  y_prob = model.predict_proba(X_test_input)
  
  # evaluate
  model_ap = average_precision_score(y_test_input, y_prob[:,1])
  model_auc = roc_auc_score(y_test_input, y_prob[:,1])
  
  # log model
  mlflow.log_metric('avg precision', model_ap)
  mlflow.log_metric('auc', model_auc)
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
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
# MAGIC The remainder of the model evaluation function is fairly straightforward.  We simply train and evaluate our model and return our loss value, *i.e.* -1 * AP Score, as part of a dictionary interpretable by hyperopt.  Based on returned values, hyperopt will generate a new set of hyperparameter values from within the search space definition with which it will attempt to improve our metric. We will limit the number of hyperopt evaluations to 250 simply based on a few trail runs we performed (not shown).  The larger the potential search space and the degree to which the model (in combination with the training dataset) responds to different hyperparameter combinations determines how many iterations are required for hyperopt to arrive at locally optimal values.  You can examine the output of the hyperopt run to see how our loss metric slowly improves over the course of each of these evaluations:

# COMMAND ----------

import mlflow

# perform evaluation
with mlflow.start_run(run_name='XGBClassifer'):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=1,                             ### INCREASE ###
    trials=SparkTrials(parallelism=1),       ### INCREASE ###
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
  params['tree_method']='hist'        # modified for CPU deployment
  params['predictor']='cpu_predictor' # modified for CPU deployment
  mlflow.log_params(params)
  
  # train
  model = Pipeline(steps=[
    ('pre', pre),
    ('clf', XGBClassifier(**params))
  ])
  model.fit(X_train_raw, y_train)
  
  # predict
  y_prob = model.predict_proba(X_test_raw)
  
  # evaluate
  model_ap = average_precision_score(y_test, y_prob[:,1])
  model_auc = roc_auc_score(y_test, y_prob[:,1])

  # log model
  wrappedModel = SklearnModelWrapper(model)
  signature = infer_signature(X_train_raw, y_prob)
  mlflow.pyfunc.log_model(python_model=wrappedModel, artifact_path='model', signature=signature)
  mlflow.log_metric('avg precision', model_ap)
  mlflow.log_metric('auc', model_auc)

  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md ## Register Champion Model

# COMMAND ----------

model_name = 'VR Fraud Analytics'

# COMMAND ----------

for i in range(3):
  fs.log_model(
    wrappedModel,
    "model",
    flavor=mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=model_name
  )

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

X_train_filter = X_train_raw[X_train_raw[['year']]==2016]

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder


# Handling numerical data
num = Pipeline(steps=[
    ('std', StandardScaler()),
    ('imp', SimpleImputer(strategy='mean'))
])

# Handling categorical data
cat = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder())
])

# Preprocessor
pre = Pipeline(steps=[(
  'preprocessor', ColumnTransformer(transformers=[
    ('num', num, numVars),
    ('cat', cat, catVars)
  ])
)])

# train
model = Pipeline(steps=[
  ('pre', pre),
  ('clf', XGBClassifier(**params, use_label_encoder=False))
])
model.fit(X_train_filter, y_train)

# predict
y_prob = model.predict_proba(X_test_raw)

# evaluate
model_ap = average_precision_score(y_test, y_prob[:,1])
model_auc = roc_auc_score(y_test, y_prob[:,1])

print(model_ap, model_auc)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

categoricals = ['bank', 'checking_savings', 'offsite_or_onsite', 'pos_capability', 'state', 'withdrawl_or_deposit']
categoricals.remove("state")
indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="skip"), categoricals))
featureCols = list(map(lambda c: c+"_idx", categoricals)) + ["amount", "year", "month", "day", "hour", "min"]

stages = indexers + [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="fraud_report", outputCol="label")]

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier 
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", seed=1028)
grid = ParamGridBuilder().addGrid(
  dt.maxDepth, [3, 4, 5]
).build()

pipeline = Pipeline(stages=stages+[dt])
tvs = TrainValidationSplit(seed=3923772, estimator=pipeline, trainRatio=0.7, evaluator=eval, estimatorParamMaps=grid)

# COMMAND ----------

model = tvs.fit(df)
# AUROC metric for the three models that were built; max depth 5 seems best:
model.validationMetrics