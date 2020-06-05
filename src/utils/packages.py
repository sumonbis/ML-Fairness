from IPython.display import Markdown, display
# numpy and pandas for data manipulation
from time import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# AIF360 Library
from aif360.datasets import *
from aif360.algorithms.preprocessing import LFR, Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

# Scikit-learn Library
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

import tensorflow as tf
