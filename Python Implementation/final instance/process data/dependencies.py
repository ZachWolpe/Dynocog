

# ========================================= Project Dependencies ========================================= #
# base modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import torch
import sys
from tqdm import tqdm
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings

# custom modules
from process_raw_data import batch_processing
from encode_processed_data import encode_data

# user settings
plt.style.use('seaborn-darkgrid')
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 4000
# ========================================= Project Dependencies ========================================= #
