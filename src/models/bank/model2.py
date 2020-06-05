#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')
from utils.packages import *
from utils.ml_fairness import *
import csv
from pathlib import Path
dir = 'bank/res/bank2/'
Path(dir).mkdir(parents=True, exist_ok=True)
f_count = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name)) and not name.startswith('.')])

fields = ['Acc', 'F1', 'DI','SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
filename = dir + str(f_count+1) + '.csv'
with open(filename, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

def fair_metrics(dataset, pred, pred_is_dataset=False):
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred

    cols = ['Accuracy', 'F1', 'DI','SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
    obj_fairness = [[1,1,1,0,0,0,0,1,0]]

    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)

        classified_metric = ClassificationMetric(dataset,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        distortion_metric = SampleDistortionMetric(dataset,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()
        f1_sc = 2 * (classified_metric.precision() * classified_metric.recall()) / (classified_metric.precision() + classified_metric.recall())

        mt = [acc, f1_sc,
                        classified_metric.disparate_impact(),
                        classified_metric.mean_difference(),
                        classified_metric.equal_opportunity_difference(),
                        classified_metric.average_odds_difference(),
                        classified_metric.error_rate_difference(),
                        metric_pred.consistency(),
                        classified_metric.theil_index()
                    ]
        w_row = []
        for i in mt:
            print("%.8f"%i)
            w_row.append("%.8f"%i)
        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(w_row)
        row = pd.DataFrame([mt],
                           columns  = cols,
                           index = [attr]
                          )
        fair_metrics = fair_metrics.append(row)

    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

    return fair_metrics

def get_fair_metrics_and_plot(data, model, plot=False, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    pred = (pred>= 0.5)*1
    fair = fair_metrics(data, pred)

    if plot:
        pass
    return fair

def get_model_performance(X_test, y_true, y_pred, probs):
    accuracy = accuracy_score(y_true, ( y_pred>= 0.5)*1)
    matrix = confusion_matrix(y_true, ( y_pred>= 0.5)*1)
    f1 = f1_score(y_true, ( y_pred>= 0.5)*1)
    return accuracy, matrix, f1

def plot_model_performance(model, X_test, y_true):
    y_pred = model.predict(X_test)
    probs = []
    accuracy, matrix, f1 = get_model_performance(X_test, y_true, y_pred, probs)


def custom_preprocessing(df):
        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0
        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        return df


dataset_orig = BankDataset(protected_attribute_names=['age'])
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()

import lightgbm as lgb
from xgboost.sklearn import XGBClassifier

lgb_train = lgb.Dataset(data=X_train, label=y_train,  free_raw_data=False)
lgb_eval = lgb.Dataset(data=X_test, label=y_test, reference=lgb_train,  free_raw_data=False)
evals_result={}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

md = XGBClassifier()
mdl = lgb.train(params,
                lgb_train,
                valid_sets = lgb_eval,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=evals_result)


plot_model_performance(mdl, X_test, y_test)

fair = get_fair_metrics_and_plot(data_orig_test, mdl)


### Reweighing
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

data_transf_train = RW.fit_transform(data_orig_train)

# Train and save the model
lgb_trainX = lgb.Dataset(data=data_transf_train.features, label=data_transf_train.labels.ravel(),  free_raw_data=False)
lgb_evalX = lgb.Dataset(data=X_test, label=y_test, reference=lgb_trainX,  free_raw_data=False)
evals_resultX={}
paramsX = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}
rf_transf = lgb.train(paramsX,
                lgb_trainX,
                valid_sets = lgb_evalX,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=evals_resultX)


data_transf_test = RW.transform(data_orig_test)
fair = get_fair_metrics_and_plot(data_transf_test, rf_transf, plot=False)


### Disperate Impact
DIR = DisparateImpactRemover()
data_transf_train = DIR.fit_transform(data_orig_train)
# Train and save the model
lgb_trainX = lgb.Dataset(data=data_transf_train.features, label=data_transf_train.labels.ravel(),  free_raw_data=False)
lgb_evalX = lgb.Dataset(data=X_test, label=y_test, reference=lgb_trainX,  free_raw_data=False)
evals_resultX={}
paramsX = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}
rf_transf = lgb.train(paramsX,
                lgb_trainX,
                valid_sets = lgb_evalX,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=evals_resultX)

#data_transf_test = DIR.transform(data_orig_test)
fair = get_fair_metrics_and_plot(data_orig_test, rf_transf, plot=False)

# Adversarial Debiasing
sess = tf.compat.v1.Session()

debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          num_epochs=10,
                          debias=True,
                          sess=sess)

# Train and save the model
debiased_model.fit(data_orig_train)

fair = get_fair_metrics_and_plot(data_orig_test, debiased_model, plot=False, model_aif=True)

#  Prejudice Remover Regularizer
debiased_model = PrejudiceRemover()

# Train and save the model
debiased_model.fit(data_orig_train)

fair = get_fair_metrics_and_plot(data_orig_test, debiased_model, plot=False, model_aif=True)
y_pred = debiased_model.predict(data_orig_test)

data_orig_test_pred = data_orig_test.copy(deepcopy=True)
# Prediction with the original RandomForest model
# Train and save the model
lgb_trainX = lgb.Dataset(data=data_transf_train.features, label=data_transf_train.labels.ravel(),  free_raw_data=False)
lgb_evalX = lgb.Dataset(data=X_test, label=y_test, reference=lgb_trainX,  free_raw_data=False)
evals_resultX={}
paramsX = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}
mdlX = lgb.train(paramsX,
                lgb_trainX,
                valid_sets = lgb_evalX,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=evals_resultX)

scores = np.zeros_like(data_orig_test.labels)
scores = mdlX.predict(data_orig_test.features).ravel().reshape(-1,1) #[:,1].reshape(-1,1)
data_orig_test_pred.scores = scores

preds = np.zeros_like(data_orig_test.labels)
preds = mdlX.predict(data_orig_test.features)
preds = (preds >= 0.5)*1
preds = preds.reshape(-1,1)
data_orig_test_pred.labels = preds

def format_probs(probs1):
    probs1 = np.array(probs1)
    probs0 = np.array(1-probs1)
    return np.concatenate((probs0, probs1), axis=1)

# Equality of Odds
EOPP = EqOddsPostprocessing(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups,
                             seed=42)
EOPP = EOPP.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred = EOPP.predict(data_orig_test_pred)

fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)

### Calibrated Equality of Odds
cost_constraint = "fnr"

CPP = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=42)

CPP = CPP.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred = CPP.predict(data_orig_test_pred)

fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)

### Reject Option Classifier
ROC = RejectOptionClassification(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups)

ROC = ROC.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred = ROC.predict(data_orig_test_pred)

fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)
print('SUCCESS: completed 1 model.')
