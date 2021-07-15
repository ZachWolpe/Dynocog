import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
import sys
from tqdm import tqdm
import pickle
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 4000

# ---- load data module ----x
import sys
sys.path.append('')
from process_raw_data import batch_processing 

class encode_data:
    def __init__(self, bp):
        self.raw = bp

        # ---- fitts ----x
        bp.fitts_data['delta']      = bp.fitts_data['fitts_prediction'] - bp.fitts_data['reaction_time_ms']
        self.fitts_summary_stats    = bp.fitts_data.groupby(['participant']).agg({
            'delta': ['mean', 'std'], 'status': ['mean', 'std']}).reset_index()

        # ---- corsi ----x 
        self.corsi_summary_stats = bp.corsi_data.groupby(['participant']).agg(
            {'highest_span': ['max'], 'n_items': ['max'], 'status': ['mean', 'std']}).reset_index()

        # ---- navon ----x 
        x = bp.navon_data
        x['correct']  = x['status'] == 1
        x['too_slow'] = x['status'] == 3
        x = x.groupby(['participant', 'level_of_target']).agg(
            {'correct': ['mean', 'std'],
            'too_slow': ['mean', 'std'],
            'reaction_time_ms': ['mean', 'std']
            })
        x = x.reset_index()
        self.navon_summary_stats = x

        # ---- nback ----x
        self.nback_summary_stats = self.raw.nback_data.groupby(['participant', 'block_number']).agg({
            'trial_counter':    ['count'],
            'score':            ['mean', 'std'],
            'status':           ['mean', 'std'],
            'miss':             ['mean', 'std'],
            'false_alarm':      ['mean', 'std'],
            'reaction_time_ms': ['mean', 'std']
        }).reset_index()



        # ------- Demographics Encoding --------x
        # q: Gender
        # - male
        # - female
        # - other
        # - prefer not to say

        # q: Handedness
        # - right
        # - left
        # - ambidextrous

        # q: What is your highest level of education?
        # - primary school
        # - high school
        # - university
        # - graduate school

        # l: income
        # q: Compared with the average, what is your income on a scale from 1 to 10 with 5 being average?
        # - {min=1,max=10,left=low,right=high,start=5}

        # l: computer_hours
        # q: How many hours do you spend playing computer games (per week)
        # - {min=0,max=100,left=low,right=high,start=0}

        df = bp.individual_data[['participant', 'participant_file', 'user_agent', 'Welcome_Screen_T', 'participant_code_a', 'feedback_T', 'age_T', 'age_a', 'gender_T', 'gender_a',
                                'handedness_T', 'handedness_a', 'education_T', 'education_a', 'income_T', 'income_a', 'income_s', 'computer_hours_T', 
                                'computer_hours_a', 'computer_hours_s']]

        # ---- extract clean data ----x
        df             = df[df['age_a'].replace(np.NaN, 'na').str.isnumeric()]          # remove nonsensical data
        df.iloc[:, 3:] = df.iloc[:, 3:].astype('float')                                 # convert to float
        df             = df[df['gender_a'].notnull()]                                   # Nan data

        # ---- create age groupings ----x
        bins            = [0, 25, 35, 45, 55, 65, 120]
        labels          = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        df['age_group'] = pd.cut(df['age_a'], bins, labels=labels, include_lowest=True)

        # ---- gender ----x
        df['gender_a'][df['gender_a'] == 1] = 'male'
        df['gender_a'][df['gender_a'] == 2] = 'female'
        df['gender_a'][df['gender_a'] == 3] = 'other'
        df['gender_a'][df['gender_a'] == 4] = 'other'

        # ---- handedness ----x
        df['handedness_a'][df['handedness_a'] == 1] = 'right'
        df['handedness_a'][df['handedness_a'] == 2] = 'left'
        df['handedness_a'][df['handedness_a'] == 3] = 'ambidextrous'

        # ---- education ----x
        df['education_a'][df['education_a'] == 1] = 'primary school'
        df['education_a'][df['education_a'] == 2] = 'high school'
        df['education_a'][df['education_a'] == 3] = 'university'
        df['education_a'][df['education_a'] == 4] = 'graduate school'

        self.demographics = df
        # ------- Demographics Encoding --------x

    
    def describe_data(self):
        """Describe the available data associated with the class"""
        message = """

        ------------------------------------------------------------------
            self.path            : raw data loc
            self.metadata        : mturk metadata
            self.mapping         : reference table
            self.data_times      : reference times table
            self.participants    : list of participant identifiers
            self.parti_code      : list of participant codes
            self.n               : total number of samples
            self.wcst_paths      : paths to wcst  raw data
            self.nback_paths     : paths to nback raw data
            self.corsi_paths     : paths to corsi raw data
            self.fitts_paths     : paths to fitts raw data
            self.navon_paths     : paths to navon raw data
            self.wcst_data       : wcst  dataframe
            self.nback_data      : nback dataframe
            self.corsi_data      : corsi dataframe
            self.fitts_data      : fitts dataframe
            self.navon_data      : navon dataframe
            self.individual_data : psytoolkit metadata
            self.MTurk           : mturk completion data

            -----------------------------------------------------
            Additions:

            self.raw                    : original object
            self.nback_summary_stats    : dataframe
            self.navon_summary_stats    : dataframe
            self.corsi_summary_stats    : dataframe
            self.fitts_summary_stats    : dataframe
            self.demographics           : dataframe
            self.plot_random_fitts      : plot
            self.plot_corsi             : plot
            self.plot_navon             : plot
            self.write_class_to_pickle  : function
            
        ------------------------------------------------------------------

        """
        print(message)


    def plot_random_fitts(self, all=False, color='lightblue'):
        x = self.raw.fitts_data
        p = np.random.choice(x.participant.unique())
        if not all:
            x.loc[x['participant'] == p,].hist('delta', bins=20, color=color)
            plt.title(f'Participant: {round(p)}')
            plt.axvline(x.loc[x['participant'] == p, ['delta']].mean()[0], color='#e390ba', linewidth=2, linestyle='--')
            plt.xlabel('Fitts Delta')
        else:       
            x.loc[:, ['delta']].hist(bins=20, color=color)
            plt.axvline(x.loc[:, ['delta']].mean()[0], color='maroon', linestyle='--')
            plt.title('All Participants')
        plt.xlabel('Fitts Delta')


    def plot_corsi(self, color='#00a0b0'):
        x = self.corsi_summary_stats
        x.highest_span.hist(label='Corsi Span', color=color)
        xbar = x['highest_span'].mean()[0]
        plt.axvline(xbar, color='#ff8cb9', linestyle='--', linewidth=2, label=f'mean: {xbar}')
        plt.legend()
        plt.title('Corsi Block Span')
        plt.ylabel('frequency')
        plt.xlabel('Corsi Block Span')
        plt.show()


    def plot_navon(self, color='#6f235f'):
        self.navon_summary_stats.hist(color=color)
        plt.tight_layout()
        plt.show()


    def pie_chart(self, dummy_var, labels, colors, title, df=None):
        if not df: df=self.demographics
        sub    = df[[dummy_var]].value_counts()
        values = sub.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig.update_traces(textfont_size=15, marker=dict(colors=colors, line=dict(color='white', width=0)))
        fig.update(layout_title_text=title)
        fig.show()
        
    def distributional_plots(self, continuous_var, cat_var, categories, labels, colors, xlab, ylab, title, df=None):
        if not df: df=self.demographics
        fig = go.Figure()
        for c in range(len(categories)):
            fig.add_trace(go.Histogram(
                x           =df[continuous_var][df[cat_var] == categories[c]],
                # histnorm    ='percent',
                name        =labels[c], 
                marker_color=colors[c],
                opacity     =1
            ))
        fig.update_layout(
            barmode         ='overlay',
            title_text      =title, 
            xaxis_title_text=xlab, 
            yaxis_title_text=ylab, 
            bargap          =0.05, 
            bargroupgap     =0.1 
        )
        fig.update_layout(barmode='group')
        fig.show()



    def write_class_to_pickle(self, path):
        """serialize object to pickle object"""

        #save it
        filename = path + 'batch_processing_object_with_encodings.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file) 

        # #load it
        # with open(filename, 'rb') as file2:
        #     bp = pickle.load(file2)
        message="""
        ------------------------------------------------------------------
        Object successfully written to path: \'{}\'!

        To retrieve run:
            with open(\'{}\', 'rb') as file2:
                bp = pickle.load(file2)
        ------------------------------------------------------------------
        """.format(filename, filename)
        print(message)

