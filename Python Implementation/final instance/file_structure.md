# Project File Structure

A brief description of the file structures used, to allow the user to efficiently navigate the _"final instance"_ section of the repo.

The sub-directories key:

 - *1-9*: _numbers_ indicate order of computation.
 -*d*: _data_ class object store.
 -*s*: _supporting_ file - testing/supporting file, non-critical & replicated elsewhere.

----------

### dashboard (_2_)

_This directory hosts the dashboard._

  - **dashapp.py**
    - 1 page, model-free analysis, dashapp script.
  - **dashboard_multitab.py**
    - Final, multitab, dashapp script.
  - **dashapp_tester.ipynb**
    - Testing environment.
  
  ##### assets
    - assets for the dashapp

----------

### data objects (_d_)

_This directory hosts the stored data class objects._

  - **batch_processing_object.pkl**
    - Batch instance of the data after cleaning the raw .txt files.
  - **batch_processing_object_with_encodings.pkl**
    - Batch instance of the data after after encoding the data (preparing for the model implementation).
  
----------

### exploratory data anaylsis (_s_)

_Supporting directory: not in direct use._

_Supporting functions and testing implementations of EDA & model free analysis - superseded by the dashapp & data processing .py files._

  - **clean_processed_data.ipynb**
    - Test file to encode demographics data, reproduced in the data processing.

  - **eda.ipynb**
    - EDA file, reproduced in the dashboard. 

  ##### participant summary stats
    - store a number of summary plots (deprecated)


----------

### model-based analysis (_3_)

_Bayesian Reinforcement learning instantiation._

----------

### model-free analysis (_s_)

_Supporting directory: not in direct use._

_Supporting functions and testing implementations of EDA & model free analysis - superseded by the dashapp & data processing .py files. Too be deprecated_

  - **Experiments_model_free_analysis.ipynb**
  - **testbed.ipynb**
  - **wcst_model_free_analysis copy.ipynb**
  - **wcst_model_free_analysis.ipynb**
  - **wcst_model_free.ipynb**

----------

### process data (_1_)

_Process the data in 3 stages:_
  1. _process raw text files._
  2. _encoded data object._
  3. _outlier detection & removal._
  3. _create supporting functions/final tables._

Each section has two notebooks:
  - _.ipynb_ file used to call the computation.
  - _.py_ file that is used to import the class objects elsewhere in the repo.

Available files:

  1. _Process raw text files_
    - Read in raw _.txt_ files, process the data & store the _batch\_processing\_data_ instance.
    - **process_raw_data.py**
    - **process_raw_data.ipynb**

  2. _Encode data_
    - Read the class prepared by `process_raw_data.py` and prepare it further for model analysis.
    - Provide data cleaning, variable selection & auxiliary functions.
    - **encode_processed_data.py**
    - **encode_processed_data.ipynb**

  3. _Outlier Detection & Removal?!?_

  4. _Produce final tables/functions/figures_
  
    - Provide a class that reads the data object produced by `encode_processed_data` that proceduces the `plotly-express` visualizations & tables utilized in the dashapp.
    - **summary_plots_and_figures.py**
