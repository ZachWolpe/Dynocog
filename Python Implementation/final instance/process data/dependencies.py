

# ========================================= Project Dependencies ========================================= #
# visualization modules
from plotly.subplots            import make_subplots
import plotly.graph_objects     as go
import matplotlib.pyplot        as plt
import plotly.express           as px

# statistical modules
from sklearn.linear_model       import RidgeCV, SGDRegressor, BayesianRidge, ElasticNetCV, LarsCV, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression
from sklearn.feature_selection  import mutual_info_classif, f_regression, mutual_info_regression, f_classif
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.base               import BaseEstimator, RegressorMixin
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.metrics            import mean_squared_error, r2_score
from sklearn.gaussian_process   import GaussianProcessRegressor
from sklearn.pipeline           import make_pipeline, Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.svm                import SVR
from statsmodels.formula.api    import ols
from mrmr                       import mrmr_regression
from scipy                      import stats
import statsmodels.api          as sm

import pandas                   as pd
import numpy                    as np
import torch


# software dev modules
from tqdm import tqdm
import warnings
import pickle
import sys
import os
import re

# custom modules
from process_raw_data import batch_processing
from encode_processed_data import encode_data
from summary_plots_and_figures import summary_plots_and_figures

# user settings
plt.style.use('seaborn-darkgrid')
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 4000
# ========================================= Project Dependencies ========================================= #



















