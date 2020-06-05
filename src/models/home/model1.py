#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
from utils.packages import *
from utils.ml_fairness import *

import csv
from pathlib import Path
dir = 'home/res/model1/'
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


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
train_control = pd.read_csv('../../dataset/home-data/application_train.csv')

features = train_control

# Extract the ids
train_ids = features['SK_ID_CURR']

# Extract the labels for training
labels = features['TARGET']

# Remove the ids and target
features = features.drop(columns = ['SK_ID_CURR']) #, 'TARGET'

le = LabelEncoder()
le_count = 0
for col in features:
    if features[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(features[col].unique())) <= 2 or col == 'CODE_GENDER':
            # Train on the training data
            le.fit(features[col])
            # Transform both training and testing data
            features[col] = le.transform(features[col])
            # Keep track of how many columns were label encoded
            le_count += 1

print('%d columns were label encoded.' % le_count)
features = pd.get_dummies(features)

# No categorical indices to record
cat_indices = 'auto'

print('Training Data Shape: ', features.shape)

# Extract feature names
feature_names = list(features.columns)
#print(feature_names)

#train_df =train_df[~train_df.isin([np.nan, np.inf, -np.inf]).any(1)]
col = list(features.columns.values)
# Convert to np arrays
features = np.array(features)
imputer = SimpleImputer(strategy = 'median')
# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))
# Fit on the training data
imputer.fit(features)
# Transform both training and testing data
features = imputer.transform(features)
features = pd.DataFrame(features, columns=col)

dataset_orig = StandardDataset(features,
                               label_name='TARGET',
                               favorable_classes=[1],
                               protected_attribute_names=['CODE_GENDER'],
                               privileged_classes=[[0]])

privileged_groups = [{'CODE_GENDER': 0}]
unprivileged_groups = [{'CODE_GENDER': 1}]

data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()

import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary',
                           class_weight = 'balanced', learning_rate = 0.05,
                           reg_alpha = 0.1, reg_lambda = 0.1,
                           subsample = 0.8, n_jobs = -1, random_state = 50, min_split_gain=.01,)

mdl = model.fit(X_train, y_train, eval_metric = 'auc', categorical_feature = cat_indices, verbose = 200)

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
