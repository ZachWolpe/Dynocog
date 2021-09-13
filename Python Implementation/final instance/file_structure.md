# Project File Structure

A brief description of the file structures used, to allow the user to efficiently navigate the _"final instance"_ section of the repo.

The sub-directories key:

 - *1-9*: _numbers_ indicate order of computation.
 -*i*: _indenpendent_ directory.
 -*d*: _data_ class object store.
 -*s*: _supporting_ file - testing/supporting file, non-critical & replicated elsewhere.

----------

### dashboard (_i_)

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

  - Supporting functions and testing implementations of EDA & model free analysis - superseded by the dashapp. To be deprecated at a future date.

  - **clean_processed_data.ipynb**
    - not in use.

  - **eda.ipynb**
    - EDA file, reproduced in the dashboard. 

  ##### participant summary stats
    - store a number of summary plots (deprecated)


----------

### model-free analysis 
  - Supporting functions and testing implementations of EDA & model free analysis - superseded by the dashapp. To be deprecated at a future date.

### process data
  - **process_raw_data.py**
    - Read in raw _.txt_ files, process the data & store the _batch\_processing\_data_ instance.
    - Process_raw_data.ipynb: testing environment.
  - **encode_processed_data.py**
    - Read the class prepared by `process_raw_data.py` and prepare it further for model analysis.
    - Provide data cleaning, variable selection & auxiliary functions.
    - encode_processed_data.ipynb: testing environment.
  - **summary_plots_and_figures.py**
    - Provide a class that reads the data object produced by `encode_processed_data` that proceduces the `plotly-express` visualizations & tables utilized in the dashapp.
