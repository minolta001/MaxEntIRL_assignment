import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from itertools import product

np.seterr(divide='ignore', invalid='ignore')

class Optimizer():
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.k = 0
        self.params = None

    def reset(self, input_params):
        self.k = 0
        self.params = input_params

    def step(self, grad, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.params += lr * grad
        self.params = self.params / self.params.sum()


def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9 
  # 9  10 11 12 13
  # 14 15 16 17 18
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a 
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))
  #trans_mat = np.zeros((25, 4, 25))
  
  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state, 
  # even itself, making it terminal, and state 24 is handled specially below.
  
  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1
      
  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1
      
  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1
      
 # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1
      
  return trans_mat


def build_state_features_gridworld():
  # There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
  # 0 0 0 0 0
  # 0 1 1 1 1
  # 0 0 2 0 0
  # 0 0 0 0 0 
  # 0 0 0 0 4
  # And the special terminal state (25) has all zero state features.

  sf = np.zeros((26,4))  
  sf[0,0] = 1
  sf[1,0] = 1
  sf[2,0] = 1
  sf[3,0] = 1
  sf[4,0] = 1
  sf[5,0] = 1
  sf[6,1] = 1
  sf[7,1] = 1
  sf[8,1] = 1
  sf[9,1] = 1
  sf[10,0] = 1
  sf[11,0] = 1
  sf[12,2] = 1
  sf[13,0] = 1
  sf[14,0] = 1
  sf[15,0] = 1
  sf[16,0] = 1
  sf[17,0] = 1
  sf[18,0] = 1
  sf[19,0] = 1
  sf[20,0] = 1
  sf[21,0] = 1
  sf[22,0] = 1
  sf[23,0] = 1
  sf[24,3] = 1
  return sf


           
def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index):
  """
  Implement steps 1-3 of Algorithm 1 in Ziebart et al.
  
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S
  term_index: the index of the special terminal state
  
  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = np.shape(trans_mat)[0]# num of states
  n_actions = np.shape(trans_mat)[1]  # num of actions
  policy = np.zeros((n_states,n_actions)) 

  zs = np.zeros(n_states)   # 26 states

  s_a_pair = [(s, a) for s, a in product(range(n_states - 1), range(n_actions))]

  zs[term_index] = 1.0    
  for _ in range (horizon):
    za = np.zeros((n_states, n_actions))
    for (s_from, a) in s_a_pair:
      for s_end in range(n_states):
        reward = state_features[s_from].dot(r_weights)
        za[s_from, a] += trans_mat[s_from, a, s_end] * np.exp(reward) * zs[s_end]
        #if(s_end == 24 and s_from == 23):
          #print(za[s_from])
      
    zs = za.sum(axis=1)
    zs[term_index] = 1.0

  policy = za / zs.reshape(-1, 1)

  return policy


  
def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Implement steps 4-6 of Algorithm 1 in Ziebart et al.
  
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s
  
  return: a size S array of expected state visitation frequencies
  """
  
  n_states = np.shape(trans_mat)[0]

  n_actions = np.shape(trans_mat)[1]
  state_freq = np.zeros(n_states)
  non_terminal_set = set(range(n_states)) - set([term_index])
  
  s_a_pair = product(non_terminal_set, range(n_actions))

  d_time = np.zeros((n_states, horizon))
  d_time[:,0] = np.copy(start_dist)

  for i in range(1, horizon):
    for s_end in range(n_states):
      for s, a in s_a_pair:
        d_time[s_end, i] = d_time[s, i - 1] * policy[s, a] * trans_mat[s, a, s_end]
  
  state_freq = np.copy(d_time.sum(axis=1))

  return state_freq
  

def calculate_feature_expectation(state_features, demos):
  n_states, n_features = state_features.shape
  f_exp = np.zeros(n_features)

  for d in demos:
    for s in d:
      f_exp += state_features[s, :]
    
  f_exp = f_exp / len(demos)
  return f_exp
      

def initialize_start_dist(n_states, demos):
  prob = np.zeros(n_states)
  
  for d in demos:
    prob[d[0]] += 1.0

  prob = prob / len(demos)
  assert(prob.sum() == 1.0)

  return prob

def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index):
  """
  Implement the outer loop of MaxEnt IRL that takes gradient steps in weight space
  
  Compute a MaxEnt reward function from demonstration trajectories
  
  trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size
  term_index: the index of the special terminal state
  
  return: a size F array of reward weights
  """
  
  n_features = np.shape(state_features)[1]
  r_weights = np.zeros(n_features)
  n_states = np.shape(trans_mat)[0]
  
  f_exp = calculate_feature_expectation(state_features, demos)
  start_dist = initialize_start_dist(n_states, demos)

  #omega = np.ones(n_features) * 1.0 
  theta = seed_weights.copy()
  delta = np.inf

  optim = Optimizer(lr=learning_rate)
  optim.reset(theta) 

  converge_threshold = 1e-4
  i = 0
  while i < n_epochs:
    #omega_old = omega.copy()
    #r_weights = state_features.dot(omega)
    r_weights = theta.copy()
    

    policy = calcMaxEntPolicy(trans_mat=trans_mat, horizon=horizon, r_weights=r_weights, state_features=state_features, term_index=term_index)
    exp_state_freq = calcExpectedStateFreq(trans_mat=trans_mat, horizon=horizon, start_dist=start_dist, policy=policy)
    
    grad = f_exp - state_features.T.dot(exp_state_freq)
    optim.step(grad)
    theta = optim.params.copy()
    #delta = np.max(np.abs(omega_old - omega)) 
    i += 1


  #r_weights = state_features.dot(omega)
  r_weights = theta.copy()
  return r_weights
  
 
 
if __name__ == '__main__':
  
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = build_state_features_gridworld() 
  demos = [[4,9,14,19,24,25],[3,8,13,18,19,24,25],[2,1,0,5,10,15,20,21,22,23,24,25],[1,0,5,10,11,16,17,22,23,24,25]]
  seed_weights = np.zeros(4)

  term_index = 25
  
  # Parameters
  n_epochs = 100
  horizon = 15
  learning_rate = 0.001
  
  # Main algorithm call
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index)
  
  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))
  print(reward_fxn)
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
  plt.show()



