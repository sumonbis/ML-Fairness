import pandas as pd
import csv
import os
import numpy as np
import sys
from aif360.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc

def fair_metrics(fname, dataset, pred, pred_is_dataset=False):
    filename = fname
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
        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]

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
        print('Computing fairness of the model.')
        for i in mt:
            #print("%.8f"%i)
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

def get_fair_metrics_and_plot(fname, data, model, plot=False, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    fair = fair_metrics(fname, data, pred)
    if plot:
        pass

    return fair

def get_model_performance(X_test, y_true, y_pred, probs):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, matrix, f1

def plot_model_performance(model, X_test, y_true):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy, matrix, f1 = get_model_performance(X_test, y_true, y_pred, probs)
