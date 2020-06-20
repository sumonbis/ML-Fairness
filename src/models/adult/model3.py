#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../utils/')
# sys.path.insert(1, '../utils/')
from packages import *
from ml_fairness import *

import csv
from pathlib import Path
dir = 'adult/res/adult3/'
Path(dir).mkdir(parents=True, exist_ok=True)
f_count = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name)) and not name.startswith('.')])

fields = ['Acc', 'F1', 'DI','SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
filename = dir + str(f_count+1) + '.csv'
with open(filename, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

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

dataset_orig = AdultDataset(protected_attribute_names=['race'],
                            privileged_classes=[[1]],
                            features_to_keep=['workclass', 'education-num', 'marital-status', 'race', 'sex', 'relationship', 'capital-gain', 'capital-loss'],
                            custom_preprocessing=custom_preprocessing)
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
mdl = model.fit(X_train, y_train)

plot_model_performance(mdl, X_test, y_test)


fair = get_fair_metrics_and_plot(filename, data_orig_test, mdl)


### Reweighing
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
data_transf_train = RW.fit_transform(data_orig_train)

# Train and save the model
rf_transf = model.fit(data_transf_train.features,
                     data_transf_train.labels.ravel())

data_transf_test = RW.transform(data_orig_test)
fair = get_fair_metrics_and_plot(filename, data_transf_test, rf_transf, plot=False)


### Disperate Impact
DIR = DisparateImpactRemover()
data_transf_train = DIR.fit_transform(data_orig_train)
# Train and save the model
rf_transf = model.fit(data_transf_train.features,
                     data_transf_train.labels.ravel())

#data_transf_test = DIR.transform(data_orig_test)
fair = get_fair_metrics_and_plot(filename, data_orig_test, rf_transf, plot=False)

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

fair = get_fair_metrics_and_plot(filename, data_orig_test, debiased_model, plot=False, model_aif=True)

#  Prejudice Remover Regularizer
debiased_model = PrejudiceRemover()

# Train and save the model
debiased_model.fit(data_orig_train)

fair = get_fair_metrics_and_plot(filename, data_orig_test, debiased_model, plot=False, model_aif=True)
y_pred = debiased_model.predict(data_orig_test)


data_orig_test_pred = data_orig_test.copy(deepcopy=True)

# Prediction with the original RandomForest model
scores = np.zeros_like(data_orig_test.labels)
scores = mdl.predict_proba(data_orig_test.features)[:,1].reshape(-1,1)
data_orig_test_pred.scores = scores

preds = np.zeros_like(data_orig_test.labels)
preds = mdl.predict(data_orig_test.features).reshape(-1,1)
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

fair = fair_metrics(filename, data_orig_test, data_transf_test_pred, pred_is_dataset=True)


### Calibrated Equality of Odds
cost_constraint = "fnr"

CPP = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=42)

CPP = CPP.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred = CPP.predict(data_orig_test_pred)

fair = fair_metrics(filename, data_orig_test, data_transf_test_pred, pred_is_dataset=True)

### Reject Option Classifier
ROC = RejectOptionClassification(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups)

ROC = ROC.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred = ROC.predict(data_orig_test_pred)

fair = fair_metrics(filename, data_orig_test, data_transf_test_pred, pred_is_dataset=True)

print('SUCCESS: completed 1 model.')
