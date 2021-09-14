# ---- load data module ----x
import sys
sys.path.append('')
from dependencies import *



class encode_data:

    def __init__(self, bp):
        # ----------- process input (bp) data -----------*
        self.raw = bp
        self.summary_table = None

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
        # additional grouping: 'block_number'
        self.nback_summary_stats = self.raw.nback_data.groupby(['participant']).agg({
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

        self.demographics_plot = df
        # --- demographics dataset: clean ---x
        df2 = df[['participant', 'age_a','gender_a','handedness_a','education_a', 'income_a', 'computer_hours_a','age_group']]
        df2_times = df[['participant', 'feedback_T', 'age_T', 'gender_T','handedness_T', 'education_T', 'income_T', 'computer_hours_T']]
        df2_times['mean_reation_time_ms'] = df2_times.iloc[:,1:].mean(axis=1)
        df2_times = df2_times[['participant', 'mean_reation_time_ms']].set_index('participant')
        self.demographics = df2.set_index('participant').join(df2_times).reset_index()
        # ------- Demographics Encoding --------x

    # ------------------------------- function: summmary dataset -------------------------------x
    def compute_summary_table(self):
        ed = self
        # ----- NBack -----x
        x = ed.nback_summary_stats.groupby('participant').agg({
            ('status', 'mean'): ['mean'],
            ('reaction_time_ms', 'mean'): ['mean']
        })
        x.columns = ['nback_status', 'nback_reaction_time_ms']
        nback = x 

        # ----- Fitts -----x
        x = ed.fitts_summary_stats[[('participant', ''), ('delta','mean')]].set_index('participant')
        x.columns = ['fitts_mean_deviation']
        fitts = x

        # ----- corsi -----x
        x = ed.corsi_summary_stats[[('participant', ''), ('highest_span','max')]].set_index('participant')
        x.columns = ['corsi_block_span']
        corsi = x

        # ----- Navon -----x
        x = ed.navon_summary_stats[[('participant', ''), ('level_of_target',''), ('correct','mean'), ('reaction_time_ms', 'mean')]].set_index('participant')
        x.columns = ['navon_level_of_target', 'navon_perc_correct', 'navon_reaction_time_ms']
        navon = x

        # ----- wcst ----x
        wcst_data = ed.raw.wcst_data.copy()
        wcst_data['correct'] = wcst_data.status==1
        wcst = wcst_data.groupby('participant').agg({
            'correct': 'mean',
            'reaction_time_ms': 'mean'
        })
        wcst.columns = ['wcst_accuracy', 'wcst_RT']

        # ---- demograpics ----x
        x = ed.demographics.set_index('participant')
        x.columns = ['demographics_' + xx for xx in x.columns]
        demo = x


        # ---- Join ----x
        df = nback
        for d in [nback, fitts, corsi, navon, wcst, demo][1:]:
            df = df.join(d, how='outer')

        # ------ discrete vars ------x
        categorical_vars = ['navon_level_of_target', 'demographics_gender_a','demographics_handedness_a', 'demographics_education_a','demographics_age_group']

        # ------ continuous vars ------x
        continuous_vars  = ['nback_status', 'nback_reaction_time_ms', 'fitts_mean_deviation', 'corsi_block_span', 'navon_perc_correct', 
                            'navon_reaction_time_ms', 'wcst_RT', 'wcst_accuracy', 'demographics_age_a', 'demographics_income_a', 
                            'demographics_computer_hours_a', 'demographics_mean_reation_time_ms']

        # ----- fix datatypes -----x
        df[categorical_vars] = df[categorical_vars].astype('string')
        df[continuous_vars]  = df[continuous_vars].astype('float')

        self.summary_table = df
    # ------------------------------- function: summmary dataset -------------------------------x
 
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
            self.summary_table          : dataframe
            self.plot_random_fitts      : plot
            self.plot_corsi             : plot
            self.plot_navon             : plot
            self.write_class_to_pickle  : function
            self.describe_data          : info
            self.clean_data_info        : info
            
        ------------------------------------------------------------------

        """
        print(message)

    def clean_data_info(self):
        """Describe the details of the summary datasets"""
        message = """

            ===========================================================================================================================
                WCST - Wisconsin Card Sorting Task                                                  DataFrame: ed.raw.wcst_date
            ---------------------------------------------------------------------------------------------------------------------------
            
                participant                     : key               : participant ID
                card_no                         : categorical       : the card shown
                correct_card                    : categorical       : the card that should be clicked of the top four on screen      
                correct_persevering             : categorical       : the card that would be clicked if the participant is persevering
                seq_no                          : numeric           : trial number
                rule                            : categorical       : matching rule  
                card_shape                      : categorical       : current card shape
                card_number                     : categorical       : current card number
                card_colour                     : categorical       : current card colour
                reaction_time_ms                : numeric           : reaction time (ms)
                status                          : categorical       : 1=correct, 2=wrong card, 3=too slow
                card_selected                   : categorical       : card chosen
                error                           : binary            : 1=error, 0=no error
                perseverance_error              : binary            : 1=perserverance error,       0=otherwise
                not_perseverance_error          : binary            : 1=not a perseveration error, 0=otherwise

            ---------------------------------------------------------------------------------------------------------------------------    
                Demographic                                                                         DataFrame: ed.demographics
            ---------------------------------------------------------------------------------------------------------------------------

                participant                     : key 
                age_a                           : numeric
                gender_a                        : categorical 
                handedness_a                    : categorical
                education_a                     : categorical
                income_a                        : categorical 
                computer_hours_a                : numeric
                age_group                       : categorical
                mean_reation_time_ms.           : numeric

            ---------------------------------------------------------------------------------------------------------------------------    
                N-Back                                                                              DataFrame: ed.nback_summary_stats
            ---------------------------------------------------------------------------------------------------------------------------

                participant                     : key 
                block_number                    : categorical   : trial block number 
                trial_counter   - count         : numeric       : number of trials in the block 
                score           - mean          : probability   : score of current trail (1=correct, 0=wrong)
                                - std           : probability       
                status          - mean          : probability   : whether the response given was a correct match (1=correct, 0=wrong)
                                - std           : probability   
                miss            - mean          : probability   : whether the response given was a miss (1=miss, 0=otherwise)
                                - std           : probability     
                false_alarm     - mean          : probability   : 1=participant clicked but there was no-match, 0=otherwise
                                - std           : probability    
                reaction_time_ms- mean          : numeric       
                                - std           : numeric  

            ---------------------------------------------------------------------------------------------------------------------------    
                Navon                                                                              DataFrame: ed.navon_summary_stats
            ---------------------------------------------------------------------------------------------------------------------------

                participant                     : key 
                level_of_target                 : categorical   : type of signal (global/local/none)
                correct         - mean          : probability   : correct action
                                - std           : probability
                too_slow        - mean          : probability   : acted too slow
                                - std           : probability
                reaction_time_ms- mean
                                - std

            ---------------------------------------------------------------------------------------------------------------------------    
                Corsi Block Span                                                                    DataFrame: ed.corsi_summary_stats
            ---------------------------------------------------------------------------------------------------------------------------

                participant                     : key 
                highest_span    - max           : numeric       : highest corsi block span
                n_items         - max           : numeric       : (max) number of items to be remembered
                status          - mean          : categorical   : current trial (1=correct, 0=wrong)
                                - std           : numerical 

            ---------------------------------------------------------------------------------------------------------------------------    
                Fitts Law                                                                          DataFrame: ed.fitts_summary_stats
            ---------------------------------------------------------------------------------------------------------------------------

                participant                     : key 
                delta           - mean          : numeric       : average deviation in expects (fitts law) performance
                                - std           : numeric       : std dev in expected (fitts law) performance
                status          - mean          : numeric       : status (1=correct, 2=error, 3=too slow)
                                - std           : numeric     
                                
            ===========================================================================================================================
        """
        print(message)

    def plot_random_fitts(self, all=False, color='lightblue'):
        # PERUSE BEFORE USE
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
        # PERUSE BEFORE USE
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
        # PERUSE BEFORE USE
        self.navon_summary_stats.hist(color=color)
        plt.tight_layout()
        plt.show()

    def pie_chart(self, dummy_var, labels, colors, title, df=None):
        # PERUSE BEFORE USE
        if not df: df=self.demographics
        sub    = df[[dummy_var]].value_counts()
        values = sub.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig.update_traces(textfont_size=15, marker=dict(colors=colors, line=dict(color='white', width=0)))
        fig.update(layout_title_text=title)
        fig.show()
        
    def distributional_plots(self, continuous_var, cat_var, categories, labels, colors, xlab, ylab, title, df=None):
        # PERUSE BEFORE USE
        if not df: df=self.demographics_plot
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

