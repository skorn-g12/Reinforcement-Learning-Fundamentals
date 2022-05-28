# RL_Basics
Caching codes related to RL

#### 3-armed bandit:
1. Epsilon-greedy.py : 
> If you run the script as is, you'll get a comparion of different epsilons' cumulative rewards vs epochs.
2. OptInitial.py :
> This script compares with optimistic intial estimate of the probablities. I've initialized each bandit's initial estimate to 10. 
3. UCB1.py :
> This script compares UCB-1, 0.1 - greedy and optimistic initial value. This might have some bug in it. I might fix this in some time

#### Grid World (3x4):
4. DP:
  > IterativePolicyEval: Implemented Bellman equation to get the value function of a policy.
  > ValueIteration: Optimized Policy Evalutaion.
5. Monte Carlo:
> Has two scripts, one to evaluate the policy(MonteCarlo-PolicyEval.py) and the other to improve the policy(MonteCarloPolicyImprovement.py)
6. Q-Learning(Q-Learning.py):
> Implements Q-learning
