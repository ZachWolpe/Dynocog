

# ========================================= Project Dependencies ========================================= #
# visualization modules
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

# statistical modules
from scipy import stats
import pandas as pd
import numpy as np
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
