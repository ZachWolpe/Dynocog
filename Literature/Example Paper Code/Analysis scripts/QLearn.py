#!/usr/bin/env python
# encoding: utf-8
#
import os, sys, datetime, pickle
import scipy as sp
import scipy.stats as stats 
import numpy as np
import matplotlib 
import matplotlib.pylab as pl
import pandas as pd
import itertools as it
import scipy.optimize as op
import seaborn as sn 
import math
from IPython import embed as shell


class QLearn(object): 
	"""Use Q-learning to find best parameter values for Beta, alphaL, alphaG given the
	observed reinforcement learning response data """ 
	
	def __init__(self, dat_train, dat_test, pMin=0.001, pMax=0.999, LLvalue=10000, npar=3):

		#start values for model optimization
		self.pMin = np.repeat(pMin,npar) 
		self.pMax = np.repeat(pMax,npar)
		self.LLvalue = LLvalue
		self.npar = npar
		
		#choice & outcome data
		self.dat_train = dat_train
		self.dat_test = dat_test


	def fit_RL_params(self, phase, start_low=0.2, start_high=0.8, step=0.6):
		""" fit_RL_params runs the Q-learning algorithm on 'train' or 'test' phase 
		data to optimize Beta, alphaG and alphaL parameters. start_params define 
		the optimization parameter space. Sequential Least Squares Programming 
		(SLSQP) optimalization is used to optimize the objective function 
		fun. args are bounded by (self.pMin, self.pMax). fit_RL_params returns the 
		maximum likelihood (-LLH) of the optimal fit. """


		start_param = np.tile(np.arange(start_low,start_high,step), (self.npar,1)) 
		start_params = list(it.product(*start_param))
		
		#calculate -LLH for train or test phase 
		for i in range(len(start_params)): 
			
			if phase == 'train': 
				LLH = self.fit_RL_train(list(start_params[i]), self.dat_train)
			else: 
				LLH = self.fit_RL_test(list(start_params[i]), self.dat_train, self.dat_test)
								
			#find lowest -LLH (=MLE)
			if LLH < self.LLvalue: 
				MLE = LLH
				self.LLvalue=LLH
				opt_start_params = start_params[i] 
				print ('%s MLE for params'%phase, start_params[i], MLE)

		##optimalisation function using opt_start_params as x0. How to return multiple variables? 			
		if phase == 'train': 				
				
			fit = op.minimize(fun=self.fit_RL_train, x0=list(opt_start_params), method='SLSQP', 
									args=(self.dat_train), bounds= zip(self.pMin, self.pMax))
			print ('-LLH: %.3f'%fit['fun'],'params:', fit['x'], 'converged:', fit['success'])	
		
		else: 
			
			fit = op.minimize(fun=self.fit_RL_test, x0=list(opt_start_params), method='SLSQP', 
								args=(self.dat_train, self.dat_test), bounds= zip(self.pMin, self.pMax))
		
		return fit


	def fit_RL_train(self, params, dat): 
		""" Performs fit on training phase RL task.   """	

		#parameters to fit
		Beta = params[0]			#inverse gain 
		alphaG = params[1]			#learning rate loss
		alphaL = params[2]			#learning rate gain

		epsilon=0.00001 			#forgetting rate?
		tau = 0 					#discounting	
		
		#recode train choice options into 0,2,4 --> 0=12/21, 2=34/43, 4=56/65 
		choices = np.copy(dat[:,0])
		for trial in zip([12,34,56], [21,43,65],[0,2,4]):
			choices[choices==trial[0]]=trial[-1]
			choices[choices==trial[1]]=trial[-1]
		
		correct = np.copy(dat[:,1])    #0=correct,1=incorrect
		reward = np.copy(dat[:,2]) 	   #0=reward,1=noreward

		#start Q-values
		prQ0=np.repeat(0.5,6) 
		prQ=prQ0

		#initialise Qvalue, probs & prediction error arrays
		QChoices = np.zeros((len(choices),2))  
		selection=np.zeros(len(choices))
		p_e=np.copy(selection)
		qvalues = np.zeros((len(choices),6))
		rs = np.copy(selection)
		q_chosen=np.copy(selection)
		q_notchosen=np.copy(selection)
		updated_qchosen = np.copy(selection)
		
		#loop over trials
		for tr in range(choices.shape[0]): 

			#calculate choice prob using soft max
			QChoice = [prQ[choices[tr]], prQ[choices[tr]+1]] 	#Qvalues of stimulus pair
			QChoices[tr]=QChoice
			
			pChoice = 1/(1+np.exp(Beta*(QChoice[1]-QChoice[0])))
			pChoice = np.array([pChoice, 1-pChoice]) 									
			pChoice = epsilon/2+(1-epsilon)*pChoice 	#choice probs of stimulus pair
	
			selection[tr] = pChoice[correct[tr]]  	#probability of the chosen stimulus

			#select correct learning rate
			if reward[tr] == 0: 
				alpha = alphaG
			elif reward[tr]==1: 
				alpha = alphaL
	
			#the q-value of the chosen stimulus, before updating
			q_chosen[tr]=prQ[choices[tr]+correct[tr]]
			q_notchosen[tr]=prQ[choices[tr]+1-correct[tr]]
			
			qvalues[tr]=prQ
			
			#update stimulus Q-value
			r=1-reward[tr] #1 or 0 
			rs[tr]=r
			prQ[choices[tr]+correct[tr]] = prQ[choices[tr]+correct[tr]] \
										+ alpha*(r-prQ[choices[tr]+correct[tr]])

			#the q-value of the chosen stimulus, after updating
			updated_qchosen[tr] = prQ[choices[tr]+correct[tr]]
						
			#calculate prediction error
			p_e[tr] = r-prQ[choices[tr]+correct[tr]]
		
			#decay all Q-values toward initial value
			prQ=prQ+tau*(0.5-prQ)
		
		loglikelihood = sum(np.log(selection)) 

		#correct for funny values
		if math.isnan(LLH): 
			loglikelihood = -1e15  
			print ('LLH is nan')
		if loglikelihood == float("inf"):
			loglikelihood = 1e15   
			print ('LLH is inf')
				
		#save model output to dataframe
		train_results = pd.DataFrame(np.array([choices, 1-correct, rs, selection, p_e, abs(p_e),
		 q_chosen, q_notchosen, updated_qchosen, QChoices[:,0]-QChoices[:,1], QChoices[:,0]+QChoices[:,1]]).T, 
		 	columns=['stim_pair', 'correct','rout', 'select_prob', 'p_e', 'abs_pe',
		 	'q_chosen', 'q_notchosen', 'updated_q', 'qdiff', 'qtotal'])
		train_Qvals = pd.DataFrame(np.hstack([qvalues, QChoices]),  
		 	columns=['qA','qB','qC','qD','qE','qF','qC1', 'qC2'])
		train_results = pd.concat([train_results, train_Qvals], axis=1)

		return train_results


class Simulate_QLearn(object): 
	"""Simulate Q-learning to validate best STAN model parameter values for Beta, alphaL, alphaG """ 
	
	def __init__(self, optimal_params):

		self.beta   = optimal_params['beta']
		self.alphaG = optimal_params['a_gain'] 
		self.alphaL = optimal_params['a_loss']
		self.epsilon= 0.00001
		self.tau	= 0


	def simulate_RL_train(self, subject_initials, paper_plot=False): 
		"""Simulate train data per run based on best fitted stan parameters to validate RL model """
		runs=[]
	
		##### SIMULATE DATA #####

		#make simulation dataset of choice options and reward -->6 runs of 60 trials 
		good = np.repeat([0,2,4], 20) #80,70,60
		bad  = np.repeat([1,3,5], 20) #20,30,40

		#good and bad choice reward probabilities --> reward=0, no reward=1
		good_prob = zip([16,14,12], [4,6,8]) 
		r_g = np.concatenate([np.concatenate([np.zeros(good_prob[x][0]),
								np.ones(good_prob[x][1])]) for x in range(len(good_prob))])
		r_b = 1-r_g 

		#simulated trial types per run 
		run = pd.DataFrame(np.array([good, bad, r_g, r_b]).T, columns=['good','bad', 'r_g', 'r_b'])
			
		#shuffle and append runs 
		run_count = [6,5][subject_initials=='s5'] #5 runs for subject's5'  
		random_seeds = range(6)
		for i in range(run_count): 
			if paper_plot: 
				#set identical random seed per run to get same simulated results for paper plots 
				run = run.sample(frac=1, random_state=random_seeds[i]).reset_index(drop=True)
			else: 
				run = run.sample(frac=1).reset_index(drop=True) 
			runs.append(run)
		
		#merge runs to one simulated session
		sim_session = pd.concat(runs, ignore_index=True)

		##### MODEL VARIABLES #####
		
		choices = np.array(sim_session['good']).astype(int)
		reward = sim_session[['r_g', 'r_b']].astype(int) # 0=reward, 1=noreward
		prQ = np.repeat(0.5,6)
		correct = np.zeros(choices.shape[0]).astype(int)
		selection = np.zeros(choices.shape[0])
		q_chosen_sim = np.zeros(choices.shape[0])
		q_unchosen_sim = np.zeros(choices.shape[0])
		rpe_sim = np.zeros(choices.shape[0])
		r=np.zeros(choices.shape[0])
		all_Qvalues = np.zeros((6, choices.shape[0]))
		QChoices = np.zeros((len(choices),2))  


		#-----------------------------------------------------------------------#
		# 				Simulate choices and choice probabilities 				#
		#-----------------------------------------------------------------------#

		for tr in range(choices.shape[0]): 

			#Qvalues stimulus pair
			QChoice = [prQ[choices[tr]], prQ[choices[tr]+1]] 
			QChoices[tr]=QChoice
					
			#Choice probabilities stimulus pair
			pChoice = 1/(1+np.exp(self.beta*(QChoice[1]-QChoice[0])))
			pChoice = np.array([pChoice, 1-pChoice]) 									
			pChoice = self.epsilon/2+(1-self.epsilon)*pChoice 

			#simulate choices based on stim choice probabilities 
			if tr == 0: 
				correct[tr] = np.random.multinomial(1, [0.5,0.5])[0]
			else: 
				correct[tr] = np.random.multinomial(1, pChoice)[0]

			#the simulated choice given the model; 0 is correct choice 
			simChoice=1-correct[tr] 

			#choice prob. given optimal params
			selection[tr]=pChoice[simChoice]
			
			#the q-value of the simulated chosen and unchosen stimulus, before updating
			q_chosen_sim[tr]=prQ[choices[tr]+simChoice]
			q_unchosen_sim[tr]=prQ[choices[tr]+1-simChoice]

			#positive learning rate
			if (simChoice==0 and reward['r_g'][tr]==0) or (simChoice==1 and reward['r_b'][tr]==0): 
				alpha = self.alphaG
			#negative learning rate 
			elif (simChoice==0 and reward['r_g'][tr]==1) or (simChoice==1 and reward['r_b'][tr]==1):
				alpha = self.alphaL
			else: 
				print('wrong reinforcement')


			#reinforcement associated with simChoice  
			if simChoice == 0: 
				r[tr]=1-reward['r_g'][tr]
			else: 
				r[tr]=1-reward['r_b'][tr]

			#calculate simulated rpe 
			rpe_sim[tr] = r[tr]-prQ[choices[tr]+simChoice]

			#update stimulus Q-value 
			prQ[choices[tr]+simChoice] = prQ[choices[tr]+simChoice] \
											+ alpha*(r[tr]-prQ[choices[tr]+simChoice])

			#decay values to initial value 
			prQ = prQ + self.tau * (0.5-prQ)
			all_Qvalues[:,tr]=prQ		
		
		#simulated results, correct simulated choice=1/incorrect=0; rewarded simulated choice=1/noreward=0
		sim_results = pd.DataFrame(np.array([choices, correct, r, selection, q_chosen_sim, 
			q_unchosen_sim, rpe_sim, QChoices[:,0]-QChoices[:,1]]).T, 
			columns=['stim_pair', 'predict','rout', 'select_prob', 'q_chosen_sim', 
			'q_unchosen_sim', 'rpe_sim', 'qdiff_sim'])
		sim_Qvals = pd.DataFrame(np.array(all_Qvalues.T), 
			columns=['sA','sB','sC','sD','sE','sF'])
		sim_results = pd.concat([sim_results, sim_Qvals], axis=1)

		return (sim_results)

