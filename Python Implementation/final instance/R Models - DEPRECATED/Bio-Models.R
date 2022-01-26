

# loc
setwd("~/Documents/Production/Dynocog/Python Implementation/final instance/R Models")

wcst       <- read.csv(file = '../model-free analysis/final_dataframes/wcst.csv')
covars <- read.csv(file = '../model-free analysis/final_dataframes/covariates.csv')



# threshold: percentage correct 
lam       = 0.70
subjects  = covars$participant[covars$wcst_accuracy > lam]

wcst_subset = wcst[wcst$participant %in% subjects,]


schools_data <- list(
  J = 8,
  y = c(28,  8, -3,  7, -1,  1, 18, 12),
  sigma = c(15, 10, 16, 11,  9, 11, 10, 18)
)

# wcst_subset.action   = wcst_subset.action.astype(int)
# wcst_subset.reward   = wcst_subset.reward.astype(int)
action_matrix = wcst_subset[['participant' ,'n_t', 'action']].pivot(index='participant', columns='n_t')
reward_matrix = wcst_subset[['participant' ,'n_t', 'reward']].pivot(index='participant', columns='n_t')
data_object = {
  'n_s':    reward_matrix.shape[0],
  'n_t':    reward_matrix.shape[1],
  'action': action_matrix,
  'reward': reward_matrix+1
}