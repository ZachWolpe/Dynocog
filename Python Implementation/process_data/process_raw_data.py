import numpy as np
import pandas as pd
import pickle
import os
import re


class batch_processing:
    """
    Input:  path to data 
    Return: tools too write pandas dataframes of data to a specified location
    """
    def __init__(self, path_to_data):
        self.path          = path_to_data
        self.mapping       = pd.read_csv(self.path + '/data.csv', index_col=False)
        self.data_times    = pd.read_csv(self.path + '/data_times.csv', index_col=False)
        self.participants  = self.mapping['participant'].tolist()
        self.parti_code    = self.mapping['participant_code:1'].tolist()
        self.n             = self.mapping.shape[0]
        self.wcst_paths    = [self.path  + wp for wp in self.mapping['wcst_task:1'].tolist()]
        self.nback_paths   = [self.path  + wp for wp in self.mapping['n_back_task:1'].tolist()]
        self.corsi_paths   = [self.path  + wp for wp in self.mapping['corsi_block_span_task:1'].tolist()]
        self.fitts_paths   = [self.path  + wp for wp in self.mapping['fitts_law:1'].tolist()]
        self.navon_paths   = [self.path  + wp for wp in self.mapping['navon_task:1'].tolist()]
        self.wcst_data     = None
        self.nback_data    = None
        self.corsi_data    = None
        self.fitts_data    = None
        self.navon_data    = None

    def create_wcst_data(self):
        message = """

        ------------------------------------------------------------------
                                WCST data created
        ------------------------------------------------------------------

        """
        print(message)
        df = pd.DataFrame()
        for p in range(self.n):
            # _____ FOR EACH PARTICIPANT -----x
            pc = self.parti_code[p]
            pt = self.participants[p]

            # _____ FOR EACH PARTICIPANT -----x
            f = open(self.wcst_paths[p], 'r')
            for l in f.readlines():
                st  = l.split(' ')
                crd = re.split(r'(\d+)', st[5]) 
                dt  = {
                    'participant':            pc,
                    'participant_code':       pt,
                    'card_no':                st[0],
                    'correct_card':           st[1],
                    'correct_persevering':    st[2],
                    'seq_no':                 st[3],
                    'rule':                   st[4],
                    'card_shape':             crd[0],
                    'card_number':            crd[1],
                    'card_colour':            crd[2],
                    'reaction_time_ms':       st[6],
                    'status':                 st[7],
                    'card_selected':          st[8],
                    'error':                  st[9],
                    'perseverance_error':     st[10],
                    'not_perseverance_error': st[11].split('\n')[0],
                }
                df = df.append(dt, ignore_index=True)[dt.keys()]
        f.close()
        self.wcst_data = df


    def create_navon_data(self):
        message = """

        ------------------------------------------------------------------
                                Navon data created
        ------------------------------------------------------------------

        """
        print(message)
        df = pd.DataFrame()
        for p in range(self.n):
            # _____ FOR EACH PARTICIPANT -----x
            pc = self.parti_code[p]
            pt = self.participants[p]

            # _____ FOR EACH PARTICIPANT -----x
            f = open(self.navon_paths[p], 'r')
            for l in f.readlines():
                st  = l.split(' ')
                dt  = {
                    'participant':            pc,
                    'participant_code':       pt,
                    'large_letter':           st[0][0],
                    'small_letter':           st[0][0],
                    'level_of_target':        st[1],
                    'level_of_target_n':      st[2],
                    'status':                 st[3],
                    'reaction_time_ms':       st[4].split('\n')[0],
                }
                df = df.append(dt, ignore_index=True)[dt.keys()]
        f.close()
        self.navon_data = df


    def create_nback_data(self):
        message = """

        ------------------------------------------------------------------
                                N back data created
        ------------------------------------------------------------------

        """
        print(message)
        df = pd.DataFrame()
        for p in range(self.n):
            # _____ FOR EACH PARTICIPANT -----x
            pc = self.parti_code[p]
            pt = self.participants[p]

            # _____ FOR EACH PARTICIPANT -----x
            f = open(self.nback_paths[p], 'r')
            for l in f.readlines():
                st  = l.split(' ')
                dt  = {
                    'participant':              pc,
                    'participant_code':         pt,
                    'block_number':             st[0],
                    'score':                    st[1],
                    'status':                   st[2],
                    'miss':                     st[3],
                    'false_alarm':              st[4],
                    'reaction_time_ms':         st[5],
                    'match':                    st[6],
                    'stimuli':                  st[7],
                    'stimuli_n_1':              st[8],
                    'stimuli_n_2':              st[9].split('\n')[0],
                }
                df = df.append(dt, ignore_index=True)[dt.keys()]
        f.close()
        self.nback_data = df


    def create_corsi_data(self):
        message = """

        ------------------------------------------------------------------
                                Corsi data created
        ------------------------------------------------------------------

        """
        print(message)
        df = pd.DataFrame()
        for p in range(self.n):
            # _____ FOR EACH PARTICIPANT -----x
            pc = self.parti_code[p]
            pt = self.participants[p]

            # _____ FOR EACH PARTICIPANT -----x
            f = open(self.corsi_paths[p], 'r')
            for l in f.readlines():
                st  = l.split(' ')
                dt  = {
                    'participant':              pc,
                    'participant_code':         pt,
                    'highest_span':             st[0],
                    'n_items':                  st[1],
                    'status':                   st[2].split('\n')[0],
                }
                df = df.append(dt, ignore_index=True)[dt.keys()]
        f.close()
        self.corsi_data = df



    def create_fitts_data(self):
        message = """

        ------------------------------------------------------------------
                                Fitts data created
        ------------------------------------------------------------------

        """
        print(message)
        df = pd.DataFrame()
        for p in range(self.n):
            # _____ FOR EACH PARTICIPANT -----x
            pc = self.parti_code[p]
            pt = self.participants[p]

            # _____ FOR EACH PARTICIPANT -----x
            f = open(self.fitts_paths[p], 'r')
            for l in f.readlines():
                st  = l.split(' ')
                dt  = {
                    'participant':              pc,
                    'participant_code':         pt,
                    'x_loc':                    st[0],
                    'y_loc':                    st[1],
                    'size':                     st[2],
                    'distance':                 st[3],
                    'fitts_prediction':         st[4],
                    'reaction_time_ms':         st[5],
                    'status':                   st[6].split('\n')[0],
                }
                df = df.append(dt, ignore_index=True)[dt.keys()]
        f.close()
        self.fitts_data = df



    def convert_data_to_int(self):
        """Change the schema of the dataframes to include integers"""
        # converter function
        def str_to_int(df, columns):
            for c in columns: df[c] = df[c].astype(int)
            return(df)

        # convert schemas
        self.fitts_data = str_to_int(self.fitts_data, 
        ['x_loc', 'y_loc', 'size', 'distance', 'fitts_prediction', 'reaction_time_ms', 'status'])

        self.corsi_data = str_to_int(self.corsi_data, ['highest_span', 'n_items', 'status'])

        self.nback_data = str_to_int(self.nback_data, 
        ['block_number', 'score', 'status','miss', 'false_alarm', 'reaction_time_ms', 'match', 
        'stimuli','stimuli_n_1', 'stimuli_n_2'])

        self.wcst_data = str_to_int(self.wcst_data, 
        ['card_no', 'correct_card', 'correct_persevering', 'seq_no', 'card_number', 'reaction_time_ms', 'status', 
        'card_selected', 'error', 'perseverance_error', 'not_perseverance_error'])

        self.navon_data = str_to_int(self.navon_data, ['level_of_target_n', 'status', 'reaction_time_ms'])
        message="""
        ------------------------------------------------------------------
        Schemas Converted!
        ------------------------------------------------------------------
        """
        print(message)


    def write_to_pickle(self, path):
        """Write the data to pickle files"""
        try: os.mkdir(path)
        except: None

        self.fitts_data.to_pickle(path + 'fitts_data.pkl')
        self.wcst_data.to_pickle(path  + 'wcst_data.pkl')
        self.nback_data.to_pickle(path + 'nback_data.pkl')
        self.corsi_data.to_pickle(path + 'corsi_data.pkl')
        self.navon_data.to_pickle(path + 'navon_data.pkl')
        message="""
        ------------------------------------------------------------------
        Dataframes successfully written to path {}!
        ------------------------------------------------------------------
        """.format(path)
        print(message)


    def read_from_pickle(self, path):
        """Read the data to pickle files"""
        self.fitts_data = pd.read_pickle(path + 'fitts_data.pkl')
        self.wcst_data  = pd.read_pickle(path + 'wcst_data.pkl')
        self.nback_data = pd.read_pickle(path + 'nback_data.pkl')
        self.corsi_data = pd.read_pickle(path + 'corsi_data.pkl')
        self.navon_data = pd.read_pickle(path + 'navon_data.pkl')
        message="""
        ------------------------------------------------------------------
        Dataframes:

            - fitts_data
            - wcst_data
            - nback_data
            - corsi_data
            - navon_data

        Successfully read from path: \'{}\'!
        ------------------------------------------------------------------
        """.format(path)
        print(message)


    def write_class_to_pickle(self, path):
        """serialize object to pickle object"""

        #save it
        filename = path + 'batch_processing_object.pkl'
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