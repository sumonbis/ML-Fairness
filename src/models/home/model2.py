#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
from utils.packages import *
from utils.ml_fairness import *
import csv
from pathlib import Path
dir = 'home/res/model2/'
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

import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def build_model_input():
    buro_bal = pd.read_csv('../../dataset/home-data/bureau_balance.csv')
    print('Buro bal shape : ', buro_bal.shape)

    buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)

    buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

    avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

    avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
    del buro_bal
    gc.collect()

    buro = pd.read_csv('../../dataset/home-data/bureau.csv')

    buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
    buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
    buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

    buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)

    del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
    gc.collect()

    buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))

    nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

    avg_buro = buro_full.groupby('SK_ID_CURR').mean()

    del buro, buro_full
    gc.collect()

    prev = pd.read_csv('../../dataset/home-data/previous_application.csv')

    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]

    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

    prev = pd.concat([prev, prev_dum], axis=1)

    del prev_dum
    gc.collect()

    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

    avg_prev = prev.groupby('SK_ID_CURR').mean()
    del prev
    gc.collect()

    pos = pd.read_csv('../../dataset/home-data/POS_CASH_balance.csv')

    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_pos = pos.groupby('SK_ID_CURR').mean()

    del pos, nb_prevs
    gc.collect()

    cc_bal = pd.read_csv('../../dataset/home-data/credit_card_balance.csv')

    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

    del cc_bal, nb_prevs
    gc.collect()

    inst = pd.read_csv('../../dataset/home-data/installments_payments.csv')
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]

    data = pd.read_csv('../../dataset/home-data/application_train.csv')
    test = pd.read_csv('../../dataset/home-data/application_test.csv')

    y = data['TARGET']
    del data['TARGET']

    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]
    categorical_feats
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])

    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    del avg_buro, avg_prev
    gc.collect()

    return data, test, y


data, test, y = build_model_input()


train_df = pd.concat([data,y], axis=1)
train_df =train_df[~train_df.isin([np.nan, np.inf, -np.inf]).any(1)]

col = list(train_df.columns.values)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(train_df)
train_df = imputer.transform(train_df)

train_df = pd.DataFrame(train_df, columns=col)

dataset_orig = StandardDataset(train_df,
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


model = LGBMClassifier(
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=30,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
)

mdl = model.fit(X_train, y_train, eval_metric= 'auc', verbose= 100)

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


# In[14]:


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
