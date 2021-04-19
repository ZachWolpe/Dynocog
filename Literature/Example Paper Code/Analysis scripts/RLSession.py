from __future__ import division, print_function

#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Joanne Van Slooten on 2016-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import os, sys, datetime, pickle, re
import math

import numpy as np
import scipy as sp
import scipy.stats as stats 
import matplotlib
import matplotlib.pyplot as pl
import pandas as pd
import bottleneck as bn
import glob
import seaborn as sn
import hedfpy
import statsmodels.api as sm
from joblib import Parallel, delayed
import itertools
import logging, logging.handlers, logging.config
import numpy.linalg as LA
from matplotlib.gridspec import GridSpec

sys.path.append(os.environ['ANALYSIS_HOME'])

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model as lm

from Tools.log import *
from Tools.other_scripts import general_funcs_jo as myfuncs
from Tools.other_scripts import functions_jw_GLM as GLM
from fir import FIRDeconvolution
from IPython import embed as shell
from QLearn import QLearn, Simulate_QLearn

class RLSession(object):
	"""Class object of the RLSession"""
	def __init__(self, subject, experiment_name, project_directory, version, aliases, pupil_hp, loggingLevel = logging.DEBUG):
		self.subject = subject
		self.experiment_name = experiment_name
		self.aliases = aliases
		self.version = version
		self.pupil_hp = pupil_hp 

		try:
			os.mkdir(os.path.join( project_directory, experiment_name ))
			os.mkdir(os.path.join( project_directory, experiment_name, self.subject.initials ))
		except OSError:
			pass
		self.project_directory = project_directory
		self.base_directory = os.path.join( self.project_directory, self.experiment_name, self.subject.initials )
		
		self.create_folder_hierarchy()
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		
		self.ho = hedfpy.HDFEyeOperator(self.hdf5_filename)
		
		# add logging for this session
		# sessions create their own logging file handler
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis in ' + self.base_directory)
		
	def create_folder_hierarchy(self):
		"""createFolderHierarchy does... guess what."""
		this_dir = self.project_directory
		for d in [self.experiment_name, self.subject.initials]:
			try:
				this_dir = os.path.join(this_dir, d)
				os.mkdir(this_dir)
			except OSError:
				pass
		for p in ['raw','processed','figs','log']:
			try:
				os.mkdir(os.path.join(self.base_directory, p))
			except OSError:
				pass
	
	def import_raw_data(self, edf_files, aliases):
		"""import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
		for edf_file, alias in zip(edf_files, aliases):
			self.logger.info('importing file ' + edf_file + ' as ' + alias)
			
			hedfpy.CommandLineOperator.ExecCommandLine('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"') )

	def import_msg_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
			
			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))
			self.ho.edf_message_data_to_hdf(alias = alias)

	def import_gaze_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
	
			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))			 
			self.ho.edf_gaze_data_to_hdf(alias = alias, 
										 pupil_hp = self.pupil_hp, 
										 pupil_lp = 4, 
										 maximal_frequency_filterbank = 0.05, 
										 minimal_frequency_filterbank = 0.002, 
										 nr_freq_bins_filterbank=20, 
										 tf_decomposition_filterbank='lp_butterworth')			
		
	def remove_HDF5(self):
		os.system('rm ' + os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5') )


	def events_and_signals_in_time(self, data_type = 'pupil_bp', requested_eye = 'L', do_plot=False, pupil_per_trial=False):				
		"""Behavioural and pupil data of the training and test phase """
				
		for alias in ['RL_train', 'RL_test']:

			#reset lists and session_time for every alias
			pupil_data = []
			pupil_baseline_data=[]
			sound_times=[]
			keypress_times=[]
			block_ommissions=[]
			gaze_x_data =[]
			gaze_x_int_data=[]
			trial_start_times=[]
			trial_end_times=[]
			stim_times=[]
			fix_times=[]
			trial_end_times_1000hz=[]
			fix_durations = []
			trial_start_EL_times =[]
			trial_end_EL_times =[]
			pupil_per_trial=[]

			session_time=0

			#concatenate pupil data runs for test and train session seperately
			if alias == 'RL_train': 

				if self.subject.initials == 's5': 
					first_trial_in_run = [0,60,120,180,240] #s5 had 5 runs in train phase
				else: 
					first_trial_in_run = [0,60,120,180,240,300] 
			else: 
				first_trial_in_run = [0,60,120,180,240] #5 runs in the test phase 
			
			run_duration = 60
			for idx, ftir in enumerate(first_trial_in_run):
				
				# load timing info per session:
				trial_times = self.ho.read_session_data(alias, 'trials')
				trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
				behavioural_data = self.ho.read_session_data(alias, 'parameters')
				sound_events_train = self.ho.read_session_data(alias, 'sounds')
				key_events_train = self.ho.read_session_data(alias, 'events')
				sound_type= np.array([behavioural_data['sound_played'] == x for x in [0,1,2]])

				session_start_EL_time = int(np.array(trial_times['trial_start_EL_timestamp'])[ftir])
				session_stop_EL_time = int(np.array(trial_times['trial_end_EL_timestamp'])[ftir+run_duration-1])
				eye_during_period = self.ho.eye_during_period([session_start_EL_time, session_stop_EL_time], alias)					
				
				pupil = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], 
					alias = alias, signal = data_type, requested_eye = eye_during_period))
				pupil_data.append((pupil - np.mean(pupil))/ pupil.std()) #z-score

				pupil_baseline = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], 
					alias = alias, signal = 'pupil_baseline', requested_eye = eye_during_period))
				pupil_baseline_data.append((pupil_baseline - np.mean(pupil_baseline))/ pupil_baseline.std())

				self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)
				
				gaze_x_int = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], 
					alias = alias, signal = 'gaze_x_int', requested_eye = eye_during_period))
				gaze_x_int_data.append(gaze_x_int)

				#concatenate event timings for each pupil block 
				trial_start_times.append(np.array(trial_times['trial_start_EL_timestamp'][ftir:ftir+run_duration] - session_start_EL_time + session_time)/self.sample_rate)
				trial_end_times.append(np.array(trial_times['trial_end_EL_timestamp'][ftir:ftir+run_duration] - session_start_EL_time + session_time)/self.sample_rate)
				trial_end_times_1000hz.append(np.array(trial_times['trial_end_EL_timestamp'][ftir:ftir+run_duration] - session_start_EL_time + session_time))
				
				sound_times.append(np.array(sound_events_train['EL_timestamp'][ftir:ftir+run_duration] - session_start_EL_time + session_time)/self.sample_rate)
				fix_times.append(np.array((trial_phase_times[trial_phase_times['trial_phase_index'] ==1]['trial_phase_EL_timestamp'])[ftir:ftir+run_duration] - session_start_EL_time + session_time) / self.sample_rate)				
				stim_times.append(np.array((trial_phase_times[trial_phase_times['trial_phase_index'] ==2]['trial_phase_EL_timestamp'])[ftir:ftir+run_duration] - session_start_EL_time + session_time) / self.sample_rate)				

				# select keypresses that fell within a single recording block. 
				all_keypress_times = np.array(key_events_train['EL_timestamp'])
				keypresses_this_run = (all_keypress_times < session_stop_EL_time) & (all_keypress_times > session_start_EL_time)
				keypress_times.append(np.array(all_keypress_times[keypresses_this_run] - session_start_EL_time + session_time)/self.sample_rate)
				

				session_time += session_stop_EL_time - session_start_EL_time

			
			#########################################
			############# TRAIN PHASE ###############
			#########################################
			
			if alias == 'RL_train': 

				#event timings 
				self.sound_times_train = np.concatenate(sound_times)
				self.pupil_data_train = np.concatenate(pupil_data)
				self.pupil_baseline_data_train = np.concatenate(pupil_baseline_data)
				self.keypress_times_train = np.concatenate(keypress_times)
				self.gaze_x_int_train = np.concatenate(gaze_x_int_data)
				self.trial_start_times_train = np.concatenate(trial_start_times)
				self.trial_end_times_train = np.concatenate(trial_end_times)
				self.trial_end_times_train_1000hz = np.concatenate(trial_end_times_1000hz)
				self.sound_type_train = sound_type
				self.stim_times_train = np.concatenate(stim_times)
				self.fix_times_train = np.concatenate(fix_times)


				#stimulus pair presentations
				current_stim_train = np.array(behavioural_data['current_stim'].astype(int))
				AB = np.array([(c_s == 12) or (c_s == 21) for c_s in current_stim_train]) #120 presentations per pair 
				CD = np.array([(c_s == 34) or (c_s == 43) for c_s in current_stim_train])
				EF = np.array([(c_s == 56) or (c_s == 65) for c_s in current_stim_train])
				self.stim_idx_train = np.vstack([AB,CD,EF])
				
				#colour-reward prob pairing for this subject
				col_prob_train = np.zeros(6)
				for i,letter in enumerate(['pair_AB_A', 'pair_AB_B', 'pair_CD_C', 'pair_CD_D', 'pair_EF_E', 'pair_EF_F']): 
					col_prob_train[i]= int(behavioural_data[letter][0])
 
				###parameter values### 
				key_responses_train = np.array(behavioural_data['key_response']) #left (-1) or right (1) response 
				optimal_choice_train = np.array(behavioural_data['optimal_response_code']) #best choice given the reward probability 
				correct_choice_train = np.array(behavioural_data['correct_response_code']) #rewarded choice 
				correct_train = np.array(behavioural_data['correct']) #received feedback 
				current_reward = np.array(behavioural_data['current_reward'])

				###parameter booleans###
				key_indices_train = np.array([(behavioural_data['key_response'] == i) for i in [-1,1]]) #left choice, right choice
				optimal_choice_indices_train = np.array([(behavioural_data['optimal_response_code'] == i) for i in [-1, 1]]) #left/right optimal response
				correct_choice_indices_train = np.array([(behavioural_data['correct_response_code'] == i) for i in [-1, 1]]) #left/right correct response 
				correct_indices_train = np.array([(behavioural_data['correct'] == i) for i in [0,1]]) #no reward, reward feedback 	
					
				#manipulation check: did feedback match with stimulus reward probability?
				rewarded_optimal_stim_choice = [np.sum(optimal_choice_train[stim] == correct_choice_train[stim]) /np.sum(AB) \
					for stim in [AB, CD, EF]]
				optimal_stim = np.array([optimal_choice_train[stim] == correct_choice_train[stim] for stim in [AB,CD,EF]])

				#rt window
				RT_train = np.array(behavioural_data['RT'])
				rt_window = np.array((RT_train > 150) & (RT_train < 3500))

				#correct responses for trials within response time window
				correct_responses = (key_responses_train == optimal_choice_train) * rt_window
				correct_stim_responses = [np.sum(correct_responses * stim)/np.sum(AB) for stim in [AB, CD, EF]]
				correct_STAN = (~correct_responses)+1 #for STAN model fit: correct =1, incorrect = 2
				
				#rewarded responses 
				rewarded_responses = key_responses_train == correct_choice_train 
				rewarded_stim_responses = np.array([np.sum(rewarded_responses * stim)/np.sum(AB) for stim in [AB, CD, EF]])

				#accumulated rewards
				accumulated_rewards = np.cumsum(current_reward)
				acc_reward_stim = pd.DataFrame(np.cumsum(np.array([current_reward[stim] for stim in [AB, CD, EF]]) \
					, axis=1).T, columns=['AB', 'CD', 'EF'])

				#correct response times 
				medianRT_stim_pairs = np.array([np.median(RT_train[stim * correct_responses]) for stim in [AB, CD, EF]])
				meanRT_stim_pairs = np.array([np.mean(RT_train[stim * correct_responses]) for stim in [AB, CD, EF]])

				#correct median and mean response times for all seperate choice options
				rt_window_pair = [(RT_train[stim]<3500) & (RT_train[stim] > 150) for stim in [AB, CD, EF]] 
				RT_optimal=[]; RT_suboptimal=[];
				for stat in [np.median, np.mean]:
					for i, cond in enumerate([AB,CD,EF]):
						RT_optimal.append(stat(RT_train[cond][optimal_stim[i] * rt_window_pair[i]]))
						RT_suboptimal.append(stat(RT_train[cond][~optimal_stim[i] * rt_window_pair[i]]))

				overview = pd.Series(np.hstack([RT_optimal, RT_suboptimal]), 
								index=['med_A', 'med_C', 'med_E', 'mea_A', 'mea_C', 'mea_E',
									    'med_B', 'med_D', 'med_F', 'mea_B', 'mea_D', 'mea_F'], 
								name=self.subject.initials)

				overview = overview.append(pd.Series(correct_stim_responses, index=['AB', 'CD', 'EF'], name=self.subject.initials))

				###### DATA FRAMES ######
				df_train_all=pd.DataFrame({'sj': self.subject.initials, 'RT': RT_train, 'feedback': correct_train, 
											'cur_stim': current_stim_train, 'correct_response': correct_choice_train, 
											'optimal_response': optimal_choice_train, 'key_response': key_responses_train,
											'correct': correct_responses, 'correct_STAN': correct_STAN,'rewarded': rewarded_responses,
											'this_reward': current_reward, 'cumsum_reward': accumulated_rewards,
											'AB': AB, 'CD': CD, 'EF': EF}) 
				df_train_all = pd.concat([df_train_all, pd.Series(~rt_window, name='ommissions')], axis=1) 
				
				#df without trials with RTs outside response time window
				df_train_valid = df_train_all[~(df_train_all['RT']>3500) & ~(df_train_all['RT'] < 150)] 
								
				print ('saving behavioral data of subject %s, session %s'%(self.subject.initials, alias))
				with pd.get_store(self.ho.input_object) as h5_file: 
					h5_file.put("/%s"%('train_behaviour/df_train_all'), df_train_all) 
					h5_file.put("/%s"%('train_behaviour/df_train_valid'), df_train_valid)
					h5_file.put("/%s"%('train_behaviour/rewarded_stim_responses'), pd.Series(np.array(rewarded_stim_responses)))
					h5_file.put("/%s"%('train_behaviour/correct_stim_responses'), pd.Series(np.array(correct_stim_responses)))
					h5_file.put("/%s"%('train_behaviour/medianRT_stim_pairs'), pd.Series(medianRT_stim_pairs))
					h5_file.put("/%s"%('train_behaviour/meanRT_stim_pairs'), pd.Series(medianRT_stim_pairs))
					h5_file.put("/%s"%('train_behaviour/overview'), overview)
					h5_file.put("/%s"%('train_behaviour/accumulated_reward'), acc_reward_stim)

				
			#########################################
			############# TEST PHASE ################ 
			#########################################

			if alias == 'RL_test': 
				
				#concatenated event timings
				self.fix_times_test = np.concatenate(fix_times)
				self.stim_times_test = np.concatenate(stim_times)
				self.sound_times_test = np.concatenate(sound_times)
				self.pupil_data_test = np.concatenate(pupil_data)
				self.keypress_times_test = np.concatenate(keypress_times)
				self.gaze_x_int_test = np.concatenate(gaze_x_int_data)
				self.trial_start_times_test = np.concatenate(trial_start_times)
				self.trial_end_times_test = np.concatenate(trial_end_times)
				self.trial_end_times_test_1000hz = np.concatenate(trial_end_times_1000hz)
				self.pupil_baseline_data_test = np.concatenate(pupil_baseline_data)

				#event indices
				self.key_indices_test = np.array([(behavioural_data['key_response'] == i) for i in [-1,1]]) #left choice, right choice
				
				#col - prob test 
				col_prob_test = np.zeros(6)
				for i,letter in enumerate(['pair_AB_A', 'pair_AB_B', 'pair_CD_C', 'pair_CD_D', 'pair_EF_E', 'pair_EF_F']): 
					col_prob_test[i]= int(behavioural_data[letter][0])

				#### parameter values ###
				cur_stim_test = np.array(behavioural_data['current_stim'].astype(int))
				key_response_test = np.array(behavioural_data['key_response'])
				correct_test = np.array(behavioural_data['correct'])
				cor_resp_code_test = np.array(behavioural_data['correct_response_code'])
				RT_test = np.array(behavioural_data['RT'])

				#choice conditions 
				win_win = [13,31,15,51,35,53]
				lose_lose = [24,42,26,62,46,64]
				win_lose_all = [12,21,14,41,16,61,32,23,34,43,36,63,52,25,54,45,56,65]
				win_lose_unique = [14,41,16,61,32,23,36,63,52,25,54,45] #not include train stimuli pairs
				approach = [13,31,14,41,15,51,16,61] #not include train stimuli pairs  
				avoid = [23,32,24,42,25,52,26,62]
				
				#boolean arrays of choice conditions
				ww_stim = np.array([cur_stim_test == w for w in win_win]).sum(axis=0, dtype=bool)
				ll_stim = np.array([cur_stim_test == l for l in lose_lose]).sum(axis=0, dtype=bool)
				wla_stim = np.array([cur_stim_test == wl for wl in win_lose_all]).sum(axis=0, dtype=bool)
				wlu_stim = np.array([cur_stim_test == wl for wl in win_lose_unique]).sum(axis=0, dtype=bool)
				ap_stim = np.array([cur_stim_test == ap for ap in approach]).sum(axis=0, dtype=bool)
				av_stim = np.array([cur_stim_test == av for av in avoid]).sum(axis=0, dtype=bool)
																	
								
				#make dataframe for behavioural data 
				df_test_all =pd.DataFrame({'sj': self.subject.initials, 'RT':RT_test, 'correct': correct_test,
										'cur_stim': cur_stim_test, 'correct_response': cor_resp_code_test, 
										'response': key_response_test, 'ww_stim': ww_stim, 'll_stim': ll_stim, 
										'wla_stim': wla_stim, 'wlu_stim': wlu_stim, 'ap_stim': ap_stim, 
										'av_stim': av_stim}) 

				#append condition labels columns to df_test_all 
				new_columns = np.tile(np.array(['xxx' for x in range(df_test_all.shape[0])]), (4,1))
				for stim in ['ww_stim', 'll_stim', 'wla_stim']: 
					new_columns[0,np.array(df_test_all[stim])] = stim		
				for stim in ['ww_stim', 'll_stim', 'wlu_stim']: 
					new_columns[1,np.array(df_test_all[stim])] = stim		
				for stim in ['ap_stim', 'av_stim']: 
					new_columns[2,np.array(df_test_all[stim])] = stim
						
				df_test_all = pd.concat([df_test_all, pd.Series(new_columns[0], name = 'stim_conds'), 
							pd.Series(new_columns[1], name='stim_condsU'), pd.Series(new_columns[2], 
							name='ap_av')], axis=1)
				
				#correct trials within response time window
				rt_window = np.array((df_test_all['RT'] > 150) & (df_test_all['RT'] < 3500))
				correct = np.array(np.array(df_test_all['correct']) * rt_window).astype(bool)  

				#append ommission column 
				df_test_all = pd.concat([df_test_all, pd.Series(~rt_window, name='ommissions')], axis=1) 
				block_ommissions = pd.Series([np.sum(RT_test[ftir:ftir+run_duration] > 3500)/run_duration for ftir in first_trial_in_run],
								 index=['B1','B2','B3','B4','B5'], name=self.subject.initials)

				#win-lose
				accuracy_win_lose = np.array([np.sum(np.array(df_test_all['stim_condsU'] == cond) * correct) / \
					np.sum(np.array(df_test_all['stim_condsU'] == cond)) for cond in ['ww_', 'wlu','ll_' ]])

				correct_rt_win_lose = np.array([df_test_all['RT'][np.array(df_test_all['stim_condsU'] == cond) * correct].mean() \
					for cond in ['ww_', 'wlu','ll_']])
				
				#approach-avoid 
				accuracy_ap_av = np.array([np.sum(np.array(df_test_all['ap_av'] == cond) * correct) / \
					np.sum(np.array(df_test_all['ap_av'] == cond)) for cond in ['ap_', 'av_']])

				correct_rt_ap_av = np.array([df_test_all['RT'][np.array(df_test_all['ap_av'] == cond) * correct].mean() \
					for cond in ['ap_','av_']])

				#all test accuracies
				test_accuracies = pd.Series({'ww_acc': accuracy_win_lose[0], 'wlu_acc': accuracy_win_lose[1],'ll_acc': accuracy_win_lose[2],  
											 'ap_acc': accuracy_ap_av[0], 'av_acc': accuracy_ap_av[1]}, 
											name=self.subject.initials)

				#all test correct RTs 
				test_correct_RTs = pd.Series({'ww_rt': correct_rt_win_lose[0], 'wlu_rt': correct_rt_win_lose[1],'ll_rt': correct_rt_win_lose[2],  
											 'ap_rt': correct_rt_ap_av[0], 'av_rt': correct_rt_ap_av[1]}, 
											name=self.subject.initials)
			
				#value difference conditions: small, medium, high 
				value_diff_cond = self.get_testphase_value_differences(df_test_all)
				correct_rt_value_diff = np.array([np.array(df_test_all['RT'])[v_d * correct] for v_d in value_diff_cond])
				accuracy_value_diff = [np.mean(correct[v_d]) for v_d in value_diff_cond]
								
				#### Save data TEST PHASE ####
				with pd.get_store(self.ho.input_object) as h5_file: 
					h5_file.put("/%s"%('test_behaviour/df_test_all'), df_test_all) 
					h5_file.put("/%s"%('test_behaviour/test_accuracies'), test_accuracies) 
					h5_file.put("/%s"%('test_behaviour/test_correct_RTs'), test_correct_RTs) 
					h5_file.put("/%s"%('test_behaviour/block_omissions'), block_ommissions)
					h5_file.put("/%s"%('test_behaviour/RT_value_difference'), pd.Series(correct_rt_value_diff))
					h5_file.put("/%s"%('test_behaviour/acc_value_difference'), pd.Series(accuracy_value_diff))
											 	

	def reverse_int(self, n):
		"""reverses the integer input """

    		return int(str(n)[::-1])


	def check_gaze_position(self, data_type='pupil_bp', do_plot=False):
		"""check whether subject's gaze position landed on coloured stimuli """

		self.events_and_signals_in_time()

		#locate gaze_x positions that fell within stimulus area (stim at 200px and 600px, border at 275 and 525 px)
		bool_gaze_stim_train = (self.gaze_x_int_train < 275) | (self.gaze_x_int_train > 525)
		bool_gaze_stim_test =  (self.gaze_x_int_test < 275)  | (self.gaze_x_int_test > 525)

		#get index gaze at stim 
		index_gaze_stim_train = [i for i, x in enumerate(bool_gaze_stim_train) if x]
		index_gaze_stim_test = [i for i, x in enumerate(bool_gaze_stim_test) if x]

		#get trial index gaze at stim 
		self.trial_gaze_at_stim_train = np.unique(np.digitize(index_gaze_stim_train, self.trial_end_times_train_1000hz))
		self.trial_gaze_at_stim_test = np.unique(np.digitize(index_gaze_stim_test, self.trial_end_times_test_1000hz))
		
		pct_gaze_at_stim_train = len(self.trial_gaze_at_stim_train)/len(self.trial_end_times_train_1000hz)*100
		pct_gaze_at_stim_test = len(self.trial_gaze_at_stim_test)/len(self.trial_end_times_test_1000hz)*100

		print('trials where gaze_x landed at stimulus area (TRAIN):\n', self.trial_gaze_at_stim_train,'\n %.2f procent \n \n'%pct_gaze_at_stim_train)
		print('trials where gaze_x landed at stimulus area (TEST):\n', self.trial_gaze_at_stim_test, '\n %.2f procent'%pct_gaze_at_stim_test)

		#save gaze_x incorrect trial indices
		with pd.get_store(self.ho.input_object) as h5_file: 
	 		h5_file.put("/%s/%s"%('gaze_at_stim', 'gaze_at_stim_train'), pd.Series(self.trial_gaze_at_stim_train))
	 		h5_file.put("/%s/%s"%('gaze_at_stim', 'gaze_at_stim_test'), pd.Series(self.trial_gaze_at_stim_test))
	 		h5_file.put("/%s/%s"%('gaze_at_stim', 'percentage_gaze_at_stim'), pd.Series([pct_gaze_at_stim_train, pct_gaze_at_stim_test]))

		if do_plot: 
			#plot gaze_x position for train and test phase 
			f = pl.figure() 
			for idx, phase in enumerate([self.gaze_x_int_train, self.gaze_x_int_test]): 
				time_points = np.linspace(0, len(phase)/1000/60, len(phase))	
				ax = f.add_subplot(2,1,idx+1)
				pl.plot(time_points, phase, alpha=0.7)
				for i,x in enumerate([200, 600, 275, 525]): 
					ax.axhline(x, color=['k', 'k', 'r', 'r'][i], lw=2,alpha=0.5)
					ax.text(0.001, x, ['stim pos L', 'stim pos R', 'gaze border L', 'gaze border R'][i], fontsize=8)
				pl.ylabel('gaze x position')
				pl.xlabel('time [s]')
			sn.despine(offset=10)
			pl.tight_layout()
			pl.savefig(os.path.join(self.base_directory, 'figs', 'gaze_x_position_train_test.pdf'))


	def get_valid_trials(self): 
		"""get_valid_trials calculates the trial indices of trials where subjects responded within the time window 
		and where there gaze did not land inside of the area of choice pairs.
		These variables are added to the dataframes df_test_all / df_train_all. 
		Also, for the train phase, it places the STAN trial-by-trial model output within df_train_all. """

		for phase in ['train', 'test']: 

			with pd.get_store(self.ho.input_object) as h5_file: 
				gaze_at_stim = np.array(h5_file.get("/%s"%('gaze_at_stim/gaze_at_stim_%s'%phase)))
				df_all_trials = h5_file.get("/%s"%('%s_behaviour/df_%s_all'%(phase, phase)))
				df_valid_trials = h5_file.get("/%s"%('%s_behaviour/df_%s_valid'%(phase, phase)))
				if phase == 'train': 
					stan_train =  h5_file.get("/%s"%('train_behaviour/STAN_variables'))

			gaze_bool = np.zeros(len(df_all_trials))
			gaze_bool[gaze_at_stim] = 1  #gaze at stim
			gaze_bool = gaze_bool.astype(bool)
			ommissions = np.array(df_all_trials['ommissions']).astype(bool) #missed trials

			#valid indices and trial numbers for ALL trials 
			self.invalid_idx_at = (np.sum([gaze_bool, ommissions], axis=0)).astype(bool)
			self.valid_idx_at = ~self.invalid_idx_at
			
			#place stan_train dataframe (based on valid trials) within dataframe of all train trials			
			if phase == 'train':
				valid_indices = np.array(df_valid_trials.index)
				for var in ['p_e', 'select_prob', 'abs_pe', 'qA', 'qB', 'qC', 'qD', 'qE', 'qF', \
							'qC1', 'qC2', 'qdiff', 'qtotal', 'q_chosen', 'q_notchosen','updated_q','stim_pair']: 
					df_all_trials[var] = np.zeros(df_all_trials.shape[0])
					df_all_trials[var].ix[valid_indices]=np.array(stan_train[var])

			#add boolean indices of valid trials (no gaze at stim, RT within response window)
			df_all_trials['valid_trials'] = self.valid_idx_at
			df_all_trials['gaze_at_stim'] = gaze_bool 

			#save df_train_all with added STAN variables 
			with pd.get_store(self.ho.input_object) as h5_file: 
					h5_file.put("/%s"%('%s_behaviour/df_%s_all'%(phase,phase)), df_all_trials) 

		
	def run_fit_RL_train_test(self, npar=3, it_to_eval=20000, do_plot=False): 
		"""fit_RL_train_test gathers and prepares train and test phase variables needed for the Q-learning algorithm. 
		behavioral variables: correct response, rewarded response, presented stimulus pair, ommission trials. 
		pupil variables: trials where eye (erroneously) landed on coloured stimulus. 
		For the Q-learning algorithm, correct and rewarded responses are recoded (correct/reward=0, 
		incorrect/no reward=1,ommission=2 """

		phase == 'train'
		with pd.get_store(self.ho.input_object) as h5_file: 
			behav_data = h5_file.get("/%s"%('%s_behaviour/df_%s_valid'%(phase, phase))) #valid trials
					
		#variables needed for Qlearning
		current_stim = np.array(behav_data['cur_stim'])
		correct = np.array(behav_data['correct'])
		reward = np.array(behav_data['rewarded'])
		
		#recode variables for STAN fit, such that 0=true and 1=false 
		correct_r = np.copy(~correct).astype(int) #1-correct
		reward_r = np.copy(~reward).astype(int)   #1-reward

		#recode train choice options into 0,2,4 for STAN fit --> 0=12/21, 2=34/43, 4=56/65      
		for trial in zip([12,34,56], [21,43,65],[0,2,4]):
			current_stim[current_stim==trial[0]]=trial[-1]
			current_stim[current_stim==trial[1]]=trial[-1]

		dat_train = np.vstack([current_stim, correct_r, reward_r]).T
			

		##### Run Qlearning model with optimal STAN parameters #####		
		optimal_param_stan_file = os.path.join(os.path.split(self.base_directory)[0],
								 'group_level', 'data', 'stan_RL_train_par_modes_%iIT_N34.pkl'%it_to_eval)				
		
		with open(optimal_param_stan_file, 'rb') as f:
			all_optimal_params = pickle.load(f) 

		
		optimal_params = all_optimal_params.ix[self.subject.initials]
		
		QL = QLearn(dat_train, dat_test, pMin=0.001, pMax=0.999, LLvalue=10000, npar=3)	
		t_res = QL.fit_RL_train(optimal_params.as_matrix(), dat_train)

		#save STAN t_res to hdf5
		with pd.get_store(self.ho.input_object) as h5_file: 
				h5_file.put("/%s"%('train_behaviour/STAN_variables'), t_res) 


	def run_simulate_RL_train(self, it_to_eval=50000, sims=100, 
								model_name ='stan_RL_train', run_count=7, conditions=3): 
		""" Run a simulation of Qlearning model with optimal stan params """

		##trials per run 
		trial_count = [360,300][self.subject.initials=='s5']
		tpr = zip(np.linspace(0, trial_count, run_count).astype(int)[:-1], 
				  np.linspace(0, trial_count, run_count).astype(int)[1:])

		##### ACTUAL BEHAVIOUR #####
		with pd.get_store(self.ho.input_object) as h5_file:
			df_train_valid = h5_file.get("/%s"%('train_behaviour/df_train_valid'))

		#valid trials per run
		vtpr = np.hstack([0, np.cumsum([sum((df_train_valid.index >= tpr[x][0]) & (df_train_valid.index < tpr[x][1]))
					 for x in range(len(tpr))])]) 
		v = zip(vtpr[:-1], vtpr[1:])
		valid_runs = [df_train_valid[v[x][0]:v[x][1]] for x in range(len(v))]
		
		#real learning across runs 
		learning_curve=np.zeros((conditions,len(tpr)))
		for idx,stim in enumerate(['AB', 'CD', 'EF']): 
			for i in range(len(valid_runs)):
				stim_per_run = valid_runs[i][stim]
				correct_stim_resp = stim_per_run & (valid_runs[i]['correct'])
				learning_curve[idx,i] = sum(correct_stim_resp)/sum(stim_per_run)				
		
		####RUN SIMULATION####
		optimal_param_stan_file = os.path.join(os.path.split(self.base_directory)[0],
								 'group_level', 'data', '%s_par_modes_%iIT_N34.pkl'
								 %(model_name, it_to_eval))		
		
		with open(optimal_param_stan_file, 'rb') as f:
			all_optimal_params = pickle.load(f) 		
		

		optimal_params = all_optimal_params.ix[self.subject.initials]							
		sim_QL = Simulate_QLearn(optimal_params)
	
		distance = 10000
		all_simulated_curves = np.zeros((sims,conditions,run_count-1))		
		best_simulated_curve = np.zeros((conditions,len(tpr)))
		
		#simulate sims times 
		for s in range(sims):
			sim_results = sim_QL.simulate_RL_train(self.subject.initials)
			#simulated learning across runs
			this_simulation=np.zeros((conditions,len(tpr)))
			for idx,stim in enumerate([0,2,4]): #a,c,e condition
				for i,t in enumerate(tpr): #runs
					stim_per_run = sim_results.ix[t[0]:t[1],'stim_pair'] == stim
					predict_per_run = (stim_per_run) & (sim_results.ix[t[0]:t[1], 'predict'])
					this_simulation[idx,i]=sum(predict_per_run)/sum(stim_per_run)            
				
				#save all simulations x runs x stimuli pair
				all_simulated_curves[s,idx] = this_simulation[idx]
					
				#distance real and sim collapsed across conditions and runs
				real_and_sim_diff = np.abs(np.mean(learning_curve - this_simulation)) 
				if real_and_sim_diff < distance:
					distance = real_and_sim_diff
					#save best simulation 
					best_simulated_curve = this_simulation		
		
		#mean across simulations		
		mean_simulated_curve = np.mean(all_simulated_curves, axis=0)

		##save real and average simulated learning curves 
		with pd.get_store(self.ho.input_object) as h5_file: 
	 		h5_file.put("/%s/%s"%('train_behaviour', 'learning_curves'), 
	 			pd.DataFrame(np.array(learning_curve).T, columns=['A','C','E']))
	 		h5_file.put("/%s/%s"%('train_behaviour', 'best_simulated_curve_%i'%(sims)), 
	 			pd.DataFrame(np.array(best_simulated_curve).T, columns=['A','C','E']))	 		
	 		h5_file.put("/%s/%s"%('train_behaviour', 'simulated_curves_%i'%(sims)), 
	 			pd.DataFrame(np.array(mean_simulated_curve).T, columns=['A','C','E']))


	def deconvolve_value_uncertainty_train_phase(self, folder, 
								data_type='pupil_bp_clean', 
								analysis_sample_rate=20, 
								interval =[-0.5, 3], 
								decision_onset=-1,
								key_onset =-1.5, 
								group='ridge_regression', 
								):
		"""disentangle value- and uncertainty-related effects on decision-related pupil dilation""" 

		self.events_and_signals_in_time(data_type = data_type)
		self.get_valid_trials()
		
		with pd.get_store(self.ho.input_object) as h5_file: 
			df_train_all = h5_file.get("/%s"%('train_behaviour/df_train_all'))
		
		RT = np.array(df_train_all['RT'])
		valid_trials = np.array(df_train_all['valid_trials'])
		correct = np.array(df_train_all['correct'])

		subsample_ratio = int(self.sample_rate/analysis_sample_rate)
		input_signal = self.pupil_data_train[::subsample_ratio]
		
		#### EVENT GAINS ####
		fix_start_idx = np.array(self.fix_times_train * analysis_sample_rate).astype(int)
		fix_events = self.fix_times_train[valid_trials]
		stim_offset_events = self.trial_end_times_train[valid_trials]

		#decision interval onset 
		all_stim_events = self.stim_times_train + decision_onset
		pos_stim_idx = all_stim_events > 0 
		stim_bools = np.array([np.array(df_train_all[stim]) for stim in ['AB', 'CD', 'EF']])
		sep_stim_events = [all_stim_events[s_b * valid_trials * pos_stim_idx] for s_b in stim_bools]
		
		#choice 
		all_key_events = self.keypress_times_train + key_onset
		pos_key_idx = all_key_events > 0 #only use positive key_events
		sep_key_events = [all_key_events[s_b * valid_trials * pos_key_idx] for s_b in stim_bools]

		#feedback 
		fb_events = self.sound_times_train[valid_trials]

		#### EVENT COVARIATES ####
		
		#RT for seperate stimulus pairs AB CD EF 
		sep_RT = [RT[s_b * valid_trials * pos_stim_idx] for s_b in stim_bools]

		q_chosen = np.array(df_train_all['q_chosen'])
		q_notchosen = np.array(df_train_all['q_notchosen'])

		#zscore across all trials...
		q_chosen_z = myfuncs.z_score(q_chosen)
		q_notchosen_z = myfuncs.z_score(q_notchosen)
		#.. and split on seperate stimulus pairs 
		q_chosen_z_split = [q_chosen_z[s_b * valid_trials * pos_stim_idx] for s_b in stim_bools] 
		q_notchosen_z_split = [q_notchosen_z[s_b * valid_trials * pos_stim_idx] for s_b in stim_bools]
		
		#zscore for all seperate stimulus pairs AB,CD,EF... 
		stim_qchosen_z = [myfuncs.z_score(q_chosen[s_b * valid_trials * pos_stim_idx]) for s_b in stim_bools]		
		stim_qnotchosen_z = [myfuncs.z_score(q_notchosen[s_b * valid_trials * pos_stim_idx]) for s_b in stim_bools]
		#... and check if q_notchosen is updated
		check_update = [np.isnan(stim_qnotchosen_z[i]).any() for i in range(len(stim_qnotchosen_z))]
		not_updated_index = np.array([i for i, x in enumerate(check_update) if x])		
		#if not updated: replace nan covariate with 0
		for i in not_updated_index:
			stim_qnotchosen_z[i] = np.zeros(len(stim_qnotchosen_z[i])) 

		rpe = np.array(df_train_all['p_e'])
		rpe_z = myfuncs.z_score(rpe)

		self.logger.info('starting deconvolution for subject %s of data_type %s in interval %s'
			%(self.subject.initials, data_type, str(interval)))
		
		events =[]
		events += fix_events, stim_offset_events,\
					sep_stim_events[0], sep_stim_events[1], sep_stim_events[2],\
					sep_key_events[0], sep_key_events[1], sep_key_events[2],\
					fb_events,  
		

		covariates = {
			'fix.gain': np.ones(len(events[0])),
			'stim_offset.gain': np.ones(len(events[1])),
			'decision_AB.gain': np.ones(len(events[2])),
			'decision_CD.gain': np.ones(len(events[3])),
			'decision_EF.gain': np.ones(len(events[4])),

			'keypress_AB.gain'	: np.ones(len(events[5])),	
			'keypress_CD.gain'	: np.ones(len(events[6])),	
			'keypress_EF.gain'	: np.ones(len(events[7])),	

			'feedback.gain': np.ones(len(events[8])),

			'decision_AB.qchosen': stim_qchosen_z[0],
			'decision_AB.q_unchosen': stim_qnotchosen_z[0],
			'decision_CD.qchosen': stim_qchosen_z[1],
			'decision_CD.q_unchosen': stim_qnotchosen_z[1],
			'decision_EF.qchosen': stim_qchosen_z[2],
			'decision_EF.q_unchosen': stim_qnotchosen_z[2],

			'keypress_AB.qchosen': stim_qchosen_z[0], 
			'keypress_AB.q_unchosen': stim_qnotchosen_z[0],
			'keypress_CD.qchosen': stim_qchosen_z[1], 
			'keypress_CD.q_unchosen': stim_qnotchosen_z[1],
			'keypress_EF.qchosen': stim_qchosen_z[2], 
			'keypress_EF.q_unchosen': stim_qnotchosen_z[2],
		
			'feedback.qchosen': q_chosen_z[valid_trials],
			'feedback.q_unchosen': q_notchosen_z[valid_trials],
		 	'feedback.RPE': rpe_z[valid_trials],
			}

		
		fd = FIRDeconvolution(
					signal = input_signal, #np.gradient(input_signal), 
					events = events,		
					event_names = ['fix', 'stim_offset', 'decision_AB', 'decision_CD', 'decision_EF',\
					 				'keypress_AB', 'keypress_CD', 'keypress_EF', 'feedback',], 
					durations = {'fix': np.ones(len(events[0]))/analysis_sample_rate,			
								'stim_offset': np.ones(len(events[1]))/analysis_sample_rate,
								'decision_AB': sep_RT[0]/self.sample_rate,
								'decision_CD': sep_RT[1]/self.sample_rate,
								'decision_EF': sep_RT[2]/self.sample_rate,
								'keypress_AB': np.ones(len(events[5]))/analysis_sample_rate,
								'keypress_CD': np.ones(len(events[6]))/analysis_sample_rate,
								'keypress_EF': np.ones(len(events[7]))/analysis_sample_rate,
								'feedback': np.ones(len(events[8]))/analysis_sample_rate},
					sample_frequency = analysis_sample_rate, 
					deconvolution_interval = interval, 
					deconvolution_frequency = analysis_sample_rate,
					covariates = covariates,
					) 	
		
		fd.create_design_matrix()

		if group == 'regression': 
			fd.regress()
		elif group == 'ridge_regression':
			fd.ridge_regress()
		else: 
			print('enter valid deconvolution method')

		fd.calculate_rsq()
		self.logger.info('r^2 for subject %s: %.2f '%(self.subject.initials, fd.rsq))

		betas = pd.DataFrame(np.zeros((fd.deconvolution_interval_size, len(fd.covariates))), columns=fd.covariates.keys())
		for i,b in enumerate(fd.covariates.keys()): 
			betas[b] = np.squeeze(fd.betas_for_cov(covariate=b))	

		folder_name = folder + '_%sHz'%analysis_sample_rate
		with pd.get_store(self.ho.input_object) as h5_file: 
	 		h5_file.put("/%s/%s"%(group, '%s_responses'%folder_name), betas)
	 		h5_file.put("/%s/%s"%(group, '%s_rsq'%folder_name), pd.Series(fd.rsq))
	 		h5_file.put("/%s/%s"%(group, '%s_tps'%folder_name), pd.DataFrame(fd.deconvolution_interval_timepoints))


	def deconvolve_train_phase(self, folder,
								data_type='pupil_bp_clean', 
								analysis_sample_rate=20, 
								interval =[-0.5, 3], 
								decision_onset=-1,
								key_onset =-2, 
								group='regression', 
								):
		"""deconvolve pupil response related to events in the learning phase """

		#import relevant variables
		self.events_and_signals_in_time(data_type = data_type)
		self.get_valid_trials()
		
		with pd.get_store(self.ho.input_object) as h5_file: 
			df_train_all = h5_file.get("/%s"%('train_behaviour/df_train_all'))

		optimal_param_stan_file = os.path.join(os.path.split(self.base_directory)[0],
								 'group_level', 'data', 'stan_RL_train_par_modes_20000IT_N34.pkl')				
		with open(optimal_param_stan_file, 'rb') as f:
			all_optimal_params = pickle.load(f) 

		#add z-scored model parameters
		cols=list(all_optimal_params.columns)
		for col in cols: 
			col_zscore = col + '_z'
			all_optimal_params[col_zscore] = (all_optimal_params[col] - all_optimal_params[col].mean()) / all_optimal_params[col].std(ddof=0)

		optimal_params = all_optimal_params.ix[self.subject.initials]

		#RT events 
		RT = np.array(df_train_all['RT'])

		#get pupil signals 
		subsample_ratio = int(self.sample_rate/analysis_sample_rate)
		input_signal = self.pupil_data_train[::subsample_ratio]
		baseline_signal = self.pupil_baseline_data_train[::subsample_ratio]  
		valid_trials = np.array(df_train_all['valid_trials'])
		correct = np.array(df_train_all['correct'])
		
		fix_start_idx = np.array(self.fix_times_train * analysis_sample_rate).astype(int)
		fix_events = self.fix_times_train[valid_trials]

		baseline_at_fix = baseline_signal[fix_start_idx]
		stim_offset_events = self.trial_end_times_train[valid_trials]
		
		all_stim_events = self.stim_times_train + decision_onset
		pos_stim_idx = all_stim_events > 0 
		stim_events = all_stim_events[pos_stim_idx * valid_trials]
		stim_bools = np.array([np.array(df_train_all[stim]) for stim in ['AB', 'CD', 'EF']])
		sep_stim_events = [all_stim_events[s_b * valid_trials * pos_stim_idx] for s_b in stim_bools]

		fb_events = self.sound_times_train[valid_trials]
		reward_events = self.sound_times_train[self.sound_type_train[1] * valid_trials]
		noreward_events = self.sound_times_train[self.sound_type_train[2] * valid_trials]
		decision_time = np.array(df_train_all['RT'])[valid_trials]/1000 #in seconds
		sep_fb_events = [self.sound_times_train[s_b * valid_trials] for s_b in stim_bools]
	
		#key_events 
		all_key_events = self.keypress_times_train + key_onset
		pos_key_events_idx = all_key_events > 0 #only use positive key_events
		key_events = all_key_events[pos_key_events_idx * valid_trials]

		#get covariates
		diffQ = np.array(df_train_all['qdiff'])
		diffQ_z = (diffQ - np.mean(diffQ))/diffQ.std()

		#totalQ 
		totalQ = np.array(df_train_all['qtotal'])
		totalQ_z = (totalQ - np.mean(totalQ))/totalQ.std()

		q_chosen = np.array(df_train_all['q_chosen'])
		q_chosen_z = (q_chosen - np.mean(q_chosen))/q_chosen.std()

		q_notchosen = np.array(df_train_all['q_notchosen'])
		q_notchosen_z = (q_notchosen - np.mean(q_notchosen))/q_notchosen.std()

		updated_q = np.array(df_train_all['updated_q'])
		updated_q_z = (updated_q - np.mean(updated_q))/updated_q.std()

		rpe = np.array(df_train_all['p_e'])
		rpe_z = (rpe - np.mean(rpe))/rpe.std()

		#signed regressor: -1 is no reward, 1 is reward
		nrpe_bool = rpe < 0 
		feedback_sign = np.ones(len(fb_events))
		feedback_sign = np.array([-1 if x else 1 for x in nrpe_bool]) 

		alpha_update = np.array([optimal_params['a_loss'] if x else optimal_params['a_gain'] for x in nrpe_bool])
		alpha_rpe = rpe * alpha_update
		alpha_rpe_z = (alpha_rpe - alpha_rpe.mean())/alpha_rpe.std()

		abs_pe = np.array(df_train_all['abs_pe'])[valid_trials]
		abs_pe_z = (abs_pe - np.mean(abs_pe))/abs_pe.std()
	
		self.logger.info('starting deconvolution for subject %s of data_type %s in interval %s '\
			%(self.subject.initials, data_type, str(interval)))
		
		events =[]
		events += fix_events, stim_offset_events, stim_events, key_events, sep_fb_events[0], \
		sep_fb_events[1], sep_fb_events[2]
		#reward_events, noreward_events #fb_events,  #stim_events, ,sep_stim_events[0], sep_stim_events[1], sep_stim_events[2],

		covariates = {
			'fix.gain': np.ones(len(events[0])),
			'stim_offset.gain': np.ones(len(events[1])),
			'decision.gain': np.ones(len(events[2])), 
			'keypress.gain'	: np.ones(len(events[3])),	
			'feedback_AB.gain': np.ones(len(events[4])),
			'feedback_CD.gain': np.ones(len(events[5])),
			'feedback_EF.gain': np.ones(len(events[6])),

			# 'reward.gain': np.ones(len(events[4])), 
			# 'noreward.gain': np.ones(len(events[5])), 

			'decision.qchosen': q_chosen_z[pos_stim_idx * valid_trials],
			'decision.q_unchosen': q_notchosen_z[pos_stim_idx * valid_trials],

			'keypress.qchosen': q_chosen_z[pos_key_events_idx * valid_trials], 
			'keypress.q_unchosen': q_notchosen_z[pos_key_events_idx * valid_trials], 

			'feedback_AB.qchosen': q_chosen_z[stim_bools[0] * valid_trials],
			'feedback_CD.qchosen': q_chosen_z[stim_bools[1] * valid_trials],
			'feedback_EF.qchosen': q_chosen_z[stim_bools[2] * valid_trials],

			'feedback_AB.qunchosen': q_notchosen_z[stim_bools[0] * valid_trials],
			'feedback_CD.qunchosen': q_notchosen_z[stim_bools[1] * valid_trials],
			'feedback_EF.qunchosen': q_notchosen_z[stim_bools[2] * valid_trials],

			#'feedback.RPE': rpe_z[valid_trials],		
			# 'reward.qchosen': q_chosen_z[self.sound_type_train[1] * valid_trials],
			# 'reward.q_unchosen': q_notchosen_z[self.sound_type_train[1] * valid_trials],
			# 'noreward.qchosen': q_chosen_z[self.sound_type_train[2]* valid_trials],
			# 'noreward.q_unchosen': q_notchosen_z[self.sound_type_train[2]*valid_trials],

			}
				
		fd = FIRDeconvolution(
					signal = input_signal, #np.gradient(input_signal), 
					events = events,		
					event_names = ['fix', 'stim_offset', 'decision', 'keypress', 
									'feedback_AB', 'feedback_CD', 'feedback_EF'], #'reward', 'noreward'
					durations = {'fix': np.ones(len(events[0]))/analysis_sample_rate,			
								'stim_offset': np.ones(len(events[1]))/analysis_sample_rate,
								'decision': (RT/1000)[pos_stim_idx * valid_trials], #duration in seconds 
								'keypress': np.ones(len(events[3]))/analysis_sample_rate,
								'feedback_AB': np.ones(len(events[4]))/analysis_sample_rate, 
								'feedback_CD':np.ones(len(events[5]))/analysis_sample_rate, 
								'feedback_EF':np.ones(len(events[6]))/analysis_sample_rate},
					sample_frequency = analysis_sample_rate, 
					deconvolution_interval = interval, 
					deconvolution_frequency = analysis_sample_rate,
					covariates = covariates,
					) 	
		
		fd.create_design_matrix()
		if group == 'regression': 
			fd.regress()
		elif group == 'ridge_regression':
			fd.ridge_regress()
		else: 
			print('enter valid deconvolution method')

		fd.calculate_rsq()
		self.logger.info('r^2 for subject %s: %.2f '%(self.subject.initials, fd.rsq))
			
		betas = pd.DataFrame(np.zeros((fd.deconvolution_interval_size, 
			len(fd.covariates))), columns=fd.covariates.keys())
		for i,b in enumerate(fd.covariates.keys()): 
			betas[b] = np.squeeze(fd.betas_for_cov(covariate=b))	

		with pd.get_store(self.ho.input_object) as h5_file: 
	 		h5_file.put("/%s/%s"%(group, '%s_responses'%folder), betas)
	 		h5_file.put("/%s/%s"%(group, '%s_rsq'%folder), pd.Series(fd.rsq))
	 		h5_file.put("/%s/%s"%(group, '%s_tps'%folder), 
	 			pd.DataFrame(fd.deconvolution_interval_timepoints))
		

	def calculate_trial_value_test_phase(self, epsilon=0.00001): 
		"""Use option Q-values at the end of the learning phase to calculate 
		Chosen and Unchosen stimulus value in the transfer phase   """
		
		self.events_and_signals_in_time()
		self.get_valid_trials()

		#model parameters 
		optimal_param_stan_file = os.path.join(os.path.split(self.base_directory)[0],
								 'group_level', 'data', 'stan_RL_train_par_modes_20000IT_N34.pkl')				
		with open(optimal_param_stan_file, 'rb') as f:
			all_optimal_params = pickle.load(f) 		
		optimal_params = all_optimal_params.ix[self.subject.initials]
		beta_param = optimal_params['beta']
		

		#get Qlearning output from training and test phase info  
		with pd.get_store(self.ho.input_object) as h5_file: 
			df_train_all  = h5_file.get("/%s"%('train_behaviour/df_train_all'))
			df_test_all  = h5_file.get("/%s"%('test_behaviour/df_test_all'))

		qtotal =  np.zeros(len(df_test_all))
		qdiff =   np.zeros(len(df_test_all))
		qchosen = np.zeros(len(df_test_all))
		qunchosen = np.zeros(len(df_test_all))
		correct_choice_prob = np.zeros(len(df_test_all))
		stim_choice_probs = np.zeros((len(df_test_all),2))

		final_qvals = np.array([df_train_all[stim].iloc[-2] for stim in ['qA', 'qB', 'qC', 'qD', 'qE', 'qF']])
		print ('subject', self.subject.initials, 'final qvals:', final_qvals)
		
		cur_stim = np.array(df_test_all['cur_stim'])
		correct = np.array(df_test_all['correct']).astype(bool)
	
		#all possible stimulus final q-value pairings (15)
		qpairings = list(itertools.combinations(final_qvals, 2))
		qnumbers = [12, 13, 14, 15, 16, 23, 24, 25, 26, 34, 35, 36, 45, 46, 56]
		
		#qvalues of correct and incorrect stimuli withinin all stimulus pairs
		correct_qs = [0,0,0,0,0,1,1,1,1,0,0,0,1,1,0]
		incorrect_qs = (~np.array(correct_qs).astype(bool)).astype(int)	
		
		q_df = pd.DataFrame(qpairings, index=qnumbers)
		q_df['correct'] = [np.array(q_df.ix[stim])[correct_qs[i]] for i, stim in enumerate(qnumbers)]
		q_df['incorrect'] = [np.array(q_df.ix[stim])[incorrect_qs[i]] for i, stim in enumerate(qnumbers)]

		#calculate  value difference and total value of presented stimuli pair	
		for i, trial_type in enumerate(cur_stim): 
			for stim in qnumbers: 
				if trial_type in [stim, self.reverse_int(stim)]: 
					qtotal[i] = sum(q_df.ix[stim])
					qdiff[i] = q_df.ix[stim][0] - q_df.ix[stim][1]
					correct_choice_prob[i] = 1/(1+np.exp(beta_param*(q_df.ix[stim]['incorrect'] - q_df.ix[stim]['correct'])))
					stim_choice_probs[i] = np.array([correct_choice_prob[i], 1-correct_choice_prob[i]]) 									
					#choice probs of stimulus pair --> first column is 'correct' stimulus choice prob
					stim_choice_probs[i] = epsilon/2+(1-epsilon)*stim_choice_probs[i] 
		qdiff = np.abs(qdiff)

		#get value of the chosen stimulus 
		for i, trial_type in enumerate(cur_stim): 
			for stim in qnumbers: 
				if trial_type in [stim, self.reverse_int(stim)]: 
					if correct[i]: 
						qchosen[i] = q_df.ix[stim]['correct']
						qunchosen[i] = q_df.ix[stim]['incorrect']
					else: 
						qchosen[i] = q_df.ix[stim]['incorrect']
						qunchosen[i] = q_df.ix[stim]['correct']

		#zscore qtotal and qdiff
		qdiff_z = (qdiff - np.mean(qdiff))/qdiff.std()
		qtotal_z = (qtotal - np.mean(qtotal))/qtotal.std()
		qchosen_z = (qchosen - np.mean(qchosen))/qchosen.std()
		qunchosen_z = (qunchosen - np.mean(qunchosen))/qunchosen.std()

		#add to df_test_all and save 
		for name, var in zip(['qdiff', 'qtotal', 'qdiff_z', 'qtotal_z', \
		 'qchosen_z', 'correct_choice_prob', 'incorrect_choice_prob', 'qunchosen_z'],
			[qdiff, qtotal, qdiff_z, qtotal_z, qchosen_z, stim_choice_probs[:,0], stim_choice_probs[:,1], qunchosen_z]): 
			df_test_all[name] = var 
		with pd.get_store(self.ho.input_object) as h5_file: 
					h5_file.put("/%s"%('test_behaviour/df_test_all'), df_test_all)


	def get_testphase_value_differences(self, df): 
		"""calculate high, medium and low value differences based on 
		presented stimuli pairs in the test phase.  
		s_diff = 10-20% diff
		m_diff = 30%-40% diff
		l_diff = 50%-60% diff """

		s_diff = np.array([24, 46, 53, 31, 26, 51, 65]) #65 is EF
		m_diff = np.array([45, 63, 25, 61, 34]) #34 is CD 
		l_diff = np.array([23, 41, 12]) #12 is AB 

		cur_stim = np.array(df['cur_stim'])
		diff_idx = np.zeros((len(df),3))
		for i,stim in enumerate(cur_stim): 		
			if (stim in s_diff) | (self.reverse_int(stim) in s_diff): 
				diff_idx[i,0]=1
			elif (stim in m_diff) | (self.reverse_int(stim) in m_diff): 
				diff_idx[i,1]=1
			else: 
				diff_idx[i,2]=1	
		value_diff_idx = diff_idx.astype(bool).T 

		return value_diff_idx 


	def deconvolve_test_phase(self, 
								group='regression', 
								folder='stimRT_qdiff',
								data_type='pupil_bp_clean_zscore', 
								analysis_sample_rate=10,  
								key_onset = -1.5,
								decision_onset=-1, 
								interval=[-0.5,3.0]): 
		"""deconvolve stim, decision and expected value variables (qtotal & qdiff) """
		self.calculate_trial_value_test_phase()

		with pd.get_store(self.ho.input_object) as h5_file: 
			df_test  = h5_file.get("/%s"%('test_behaviour/df_test_all'))
		valid_trials= np.array(df_test['valid_trials'])
		correct = np.array(df_test['correct']).astype(bool)

		subsample_ratio = int(self.sample_rate/analysis_sample_rate)
		input_signal = self.pupil_data_test[::subsample_ratio]
		baseline_signal = self.pupil_baseline_data_test[::subsample_ratio]  

		fix_start_idx = np.array(self.fix_times_test * analysis_sample_rate).astype(int)
		fix_events = self.fix_times_test[valid_trials]
		baseline_at_fix = baseline_signal[fix_start_idx]
		stim_offset_events = self.trial_end_times_test[valid_trials]

		#stimulus events 
		all_stim_events = self.stim_times_test + decision_onset
		pos_stim_events_idx = all_stim_events > 0 
		stim_events = all_stim_events[pos_stim_events_idx * valid_trials]

		all_key_events = self.keypress_times_test + key_onset
		pos_key_events_idx = all_key_events > 0 #only use positive key_events
		key_events = all_key_events[valid_trials * pos_key_events_idx]

		RT = np.array(df_test['RT'])/1000 #in seconds

		#calculate value difference between stimuli 
		value_diff_idx = self.get_testphase_value_differences(df_test)
		stim_events_value_diff = np.array([all_stim_events[v_d * valid_trials * pos_stim_events_idx] 
			for v_d in value_diff_idx])
		key_events_value_diff = np.array([all_key_events[v_d * valid_trials * pos_key_events_idx] 
			for v_d in value_diff_idx])
		RT_value_diff = np.array([RT[v_d * valid_trials * pos_stim_events_idx] 
			for v_d in value_diff_idx])

		#covariates
		qtotal_z = np.array(df_test['qtotal_z'])/5.0
		qdiff_z = np.array(df_test['qdiff_z'])/5.0
		qchosen_z = np.array(df_test['qchosen_z'])/5.0
		q_unchosen_z = np.array(df_test['qunchosen_z'])/5.0

		correct_choice_prob = np.array(df_test['correct_choice_prob'] - df_test['incorrect_choice_prob'])
		correct_choice_prob_z = (correct_choice_prob - np.mean(correct_choice_prob))/np.std(correct_choice_prob)

		self.logger.info('starting test phase deconvolution for subject %s of data_type %s in interval %s '
			%(self.subject.initials, data_type, str(interval)))
		
		events=[]
		events += fix_events, stim_offset_events, stim_events_value_diff[0], stim_events_value_diff[1], stim_events_value_diff[2], key_events_value_diff[0], key_events_value_diff[1],  key_events_value_diff[2]    
	
		
		covariates = {
			'fix.gain': np.ones(len(events[0])),
			#'stim.gain': np.ones(len(events[1])),
			'stim_offset.gain': np.ones(len(events[1])),
			'decision_s.gain': np.ones(len(events[2])),	
			'decision_m.gain': np.ones(len(events[3])),	
			'decision_l.gain': np.ones(len(events[4])),	

			'keypress_s.gain'	: np.ones(len(events[5])),
			'keypress_m.gain'	: np.ones(len(events[6])),
			'keypress_l.gain'	: np.ones(len(events[7])),

			# 'decision.qchosen':qchosen_z[valid_trials * pos_stim_events_idx], 
			# 'decision.qunchosen': q_unchosen_z[valid_trials * pos_stim_events_idx],
			# 'decision.qtotal': qtotal_z[valid_trials * pos_stim_events_idx],

			# 'keypress.qchosen':qchosen_z[valid_trials * pos_key_events_idx], 
			# 'keypress.qunchosen':q_unchosen_z[valid_trials * pos_key_events_idx], 
			# 'keypress.qtotal':qtotal_z[valid_trials * pos_key_events_idx], 			
			}
	
		fd = FIRDeconvolution(
					signal = input_signal, 
					events = events,
					event_names = ['fix','stim_offset', 'decision_s', 'decision_m', 'decision_l', 'keypress_s', 
					'keypress_m', 'keypress_l'], 
					durations = {'fix': np.ones(len(events[0]))/analysis_sample_rate,				
					'stim_offset': np.ones(len(events[1]))/analysis_sample_rate,	
					'decision_s': RT_value_diff[0],
					'decision_m': RT_value_diff[1],
					'decision_l': RT_value_diff[2],
					'keypress_s': np.ones(len(events[5]))/analysis_sample_rate,
					'keypress_m': np.ones(len(events[6]))/analysis_sample_rate,
					'keypress_l': np.ones(len(events[7]))/analysis_sample_rate,
					},
					sample_frequency = analysis_sample_rate, 
					deconvolution_interval = interval, 
					deconvolution_frequency = analysis_sample_rate,
					covariates = covariates,
					) 			
		fd.create_design_matrix()

		if group == 'regression_test': 
			fd.regress()
		elif group == 'ridge_regression_test': 
			fd.ridge_regress()	
		fd.calculate_rsq()

		self.logger.info('r^2 for subject %s: %.2f '%(self.subject.initials, fd.rsq))

		betas = pd.DataFrame(np.zeros((fd.deconvolution_interval_size, len(fd.covariates))), columns=fd.covariates.keys())
		for i,b in enumerate(fd.covariates.keys()): 
			betas[b] = np.squeeze(fd.betas_for_cov(covariate=b))	
		
		with pd.get_store(self.ho.input_object) as h5_file: 
	 		h5_file.put("/%s/%s"%(group, '%s_responses'%folder), betas)
	 		h5_file.put("/%s/%s"%(group, '%s_gof'%folder), pd.Series(fd.rsq))
	 		h5_file.put("/%s/%s"%(group, '%s_tps'%folder), pd.DataFrame(fd.deconvolution_interval_timepoints))




