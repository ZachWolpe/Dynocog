def gather_dataframes_from_hdfs(self, group = 'deconvolve_colour_sound', data_type = 'time_points'): 
	"""gather_data_from_hdfs takes group/datatype data from hdf5 files for all self.sessions.
	arguments:  group - the folder in the hdf5 file from which to take data
				data_type - the type of data to be read.
	returns a pd.dataFrame with a hierarchical index, meaning that you can access a specific self.session using 
	its key. Keys are converted from strings to numbers
	"""
			
	gathered_dataframes = [] 		
	for s in self.sessions: 
		with pd.get_store(s.hdf5_filename) as h5_file:	
			gathered_dataframes.append(pd.DataFrame(h5_file.get("/%s/%s"%(group, data_type))))	
	return gathered_dataframes



def run_pystan_RL_train(self, iterations=2000, chains=4, model_name='stan_RL_train'): 
	"""run pySTAN on reinforcement learning data to estimate beta, alpha loss, alpha gain parameter """

	#n_s 		= number of subjects 
	#n_t 		= number of trials 
	#correct 	= correct response; 1=correct/2=incorrect
	#reward     = rewarded response;0=no reward/1=reward
	#cur_stim	= presented stimuli per trial
	#init_sj	= marks start of a new subject 
	#subject	= mapping between trial and subject nr 

	#STAN model
	stan_model_code = stan_model.import_stan_model_code()

	#behavioural data train phase (all trials within response window <3500ms)
	df_train_sjs = self.gather_dataframes_from_hdfs(group = 'train_behaviour', data_type = 'df_train_valid') 
	len_valid = [len(v) for v in df_train_sjs]

	n_s = len(self.sessions) 
	n_t = [len(df_train_sjs[s]['cur_stim']) for s in range(len(self.sessions))]
	total_n_t = np.sum(n_t)

	#concatenate all relevant behavioural data across subjects 
	correct = np.concatenate([np.array(df_train_sjs[s]['correct_STAN']) for s in range(len(self.sessions))])
	reward = np.concatenate([np.array(df_train_sjs[s]['rewarded']).astype(int) for s in range(len(self.sessions))])
	cur_stim = np.concatenate([np.array(df_train_sjs[s]['cur_stim']) for s in range(len(self.sessions))])
	init_sj = np.concatenate([np.r_[1, np.zeros(n_t[s]-1)].astype(int) for s in range(len(self.sessions))])
	subject = np.concatenate([np.repeat(s+1, n_t[s]) for s in range(len(self.sessions))])

	#recode cur_stim into 1,3,5 to sync with STAN 
	for trial in zip([12,34,56], [21,43,65],[1,3,5]):
		cur_stim[cur_stim==trial[0]]=trial[-1]
		cur_stim[cur_stim==trial[1]]=trial[-1]

	#data to feed into the model
	stan_data = {'n_s': n_s, 'n_t': total_n_t, 'Choice': cur_stim, 'Correct': correct, 
				'Reward': reward, 'Init': init_sj, 'Subject': subject}

	#initialise stan model 
	model_pkl = os.path.join(self.grouplvl_data_dir, 'models', 'model_%s.pkl')%model_name	
	check = os.path.isfile(model_pkl)

	if check == False:  
		model = pystan.StanModel(model_code=stan_model_code, model_name=model_name)	#compile model			 
		shortened = model.model_name.split('_')[:-1] #save shortened model name 
		shortened_name = "_".join(shortened)
		model_pkl = os.path.join(self.grouplvl_data_dir, 'models', 'model_%s.pkl'%shortened_name)
	else:
		with open(model_pkl, 'rb') as f: 
			model = pickle.load(f)

	
	#fit stan model 
	fit = model.sampling(data=stan_data, 
					  	 iter=iterations, 
					  	 chains=chains, 
					  	 n_jobs=-1)		
	
	#folder to save fitted results
	stan_pkl = os.path.join(self.grouplvl_data_dir, \
		'%s_IT%i_CH%i_N%i.pkl'%(model_name, iterations,chains,len(self.sessions)))
	stan_ext_pkl = os.path.join(self.grouplvl_data_dir, \
		'extended_%s_IT%i_CH%i_N%i.pkl'%(model_name, iterations,chains,len(self.sessions)))				
	stan_varname_pkl = os.path.join(self.grouplvl_data_dir, \
		'varnames_%s_IT%i_CH%i_N%i.pkl'%(model_name, iterations,chains,len(self.sessions)))				
	
	for file,content in zip([model_pkl, stan_pkl, stan_ext_pkl, stan_varname_pkl], 
							[model, fit.extract(permuted=True), fit.extract(permuted=False), 
							 fit.flatnames]):
		with open(file, 'wb') as f:
			pickle.dump(content, f)

	#group parameters 
	f = pl.figure() 
	fit.traceplot(pars={'mu_b','mu_ag','mu_al'}) 
	pl.tight_layout()
	sn.despine(offset=10)
	pl.savefig(os.path.join(self.grouplvl_plot_dir, 'STAN', 'stan_group_params_%s_%sIT_N=%i.pdf'\
		%(model_name, iterations, len(self.sessions))))

	#evaluate model and extract individual parameter modes. 
	self.evaluate_pystan_RL_train(it_to_eval=iterations, 
									chains_to_eval=chains, 
									calculate_modes=True, 
									model_name=model_name)


def evaluate_pystan_RL_train(self, it_to_eval=10000, 
							chains_to_eval=4, calculate_modes=False, 
							model_name = 'stan_RL_train'): 
	"""evaluate the fitted stan_RL_train model """

	names = [s.subject.initials for s in self.sessions]
	#open fitted model to evaluate		
	stan_pkl = os.path.join(self.grouplvl_data_dir, '%s_IT%i_CH%i_N%i.pkl'
		%(model_name, it_to_eval,chains_to_eval,len(self.sessions)))				
	with open(stan_pkl, 'rb') as f:
		fit = pickle.load(f)

	greys=sn.color_palette("Greys",3)
	cols = ["#a00498", "#fec615", "#0a888a",] #purple, yellow, cyan 
	
	#group level probablity density plot
	f = pl.figure(figsize=(2.4,3))
	ds_mu_b = fit['mu_b']/100  #beta/100 for visualization
	sn.kdeplot(ds_mu_b, label=True, shade=True, color=cols[0])
	for i, p in enumerate(['mu_ag', 'mu_al']): 
		sn.kdeplot(fit[p], label=True, shade=True, color=[cols[1], cols[2]][i])
	pl.legend(['beta', 'a_gain', 'a_loss'])
	pl.ylabel('Density')
	pl.xlim([0, 0.25])
	pl.xticks([0.00, 0.10, 0.20])
	pl.title('Learning: \nParameters')
	pl.tight_layout()
	sn.despine(offset=10)
	pl.savefig(os.path.join(self.grouplvl_plot_dir, 'STAN', 'stan_group_probability_density_plot_%s_%sIT_N=%i.pdf'%(model_name, it_to_eval, len(self.sessions))))
						
	#calculate b, ag, al parameter modes for each subject
	if calculate_modes: 	
		modes = np.zeros((len(self.sessions),3))
		scales = np.zeros((len(self.sessions),3))
		for i,param in enumerate(['b_ind','ag_ind','al_ind']): 	
			for sj in range(len(self.sessions)): 			
				loc_param, scale_param = stats.norm.fit(fit[param][:,sj])
				modes[sj,i] = loc_param 
				scales[sj,i] = scale_param
		par_modes = pd.DataFrame(modes, columns=['beta', 'a_gain', 'a_loss'], index=names)
		par_scales = pd.DataFrame(scales, columns=['beta', 'a_gain', 'a_loss'], index=names)

		#save parameter modes and scales (sd)
		par_mode_pkl = os.path.join(self.grouplvl_data_dir,'%s_par_modes_%iIT_N%i.pkl'
			%(model_name, it_to_eval, len(self.sessions)))		
		par_scale_pkl = os.path.join(self.grouplvl_data_dir,'%s_par_scales_%iIT_N%i.pkl'
			%(model_name, it_to_eval, len(self.sessions)))					
		for file, content in zip([par_mode_pkl, par_scale_pkl], 
								 [par_modes, par_scales]):
			with open(file, 'wb') as f: 
				pickle.dump(content, f)

	#individual parameter estimates
	sn.set(font_scale=0.7, style="ticks")
	for var in ['b_ind', 'ag_ind', 'al_ind']: 
		fig, axes = pl.subplots(nrows=5, ncols=7, sharey=False, sharex=False, figsize=(12,9))       
		for i, ax in enumerate(axes.flat[:fit[var].shape[-1]]):
			sn.kdeplot(fit[var][:,i], ax=ax)
			ax.set_title('%s'%names[i])
		axes[2,0].set_ylabel('Frequency')
		axes[4,3].set_xlabel('%s value'%var)
		pl.tight_layout()
		sn.despine(offset=5)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'STAN', '%s_individual_%s_param_%iIT_N%i.pdf'
			%(model_name, var, it_to_eval, len(self.sessions))))

