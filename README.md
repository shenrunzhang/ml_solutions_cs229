# CS229 
My own solutions to Stanford's CS 229 machine learning course problem sets, including coding projects. Here are some highlights.

## Cart-Pole Balancing

(aka Inverted Pendulum Problem)

![](https://github.com/shenrunzhang/ml_solutions_cs229/blob/main/problemset_solutions/code/cartpole_qlearning/result.gif)

This project involves using reinforcement learning to solve the inverted pendulum problem through Markov decision processes. The problem is formulated as a cart-pole system, where the objective is to balance a pole on a cart by controlling the movement of the cart. The code discretizes the state space, and solves for the optimum policy through value and policy iteration.

The system is determined to have failed if the pole falls over or if the cart has moved beyond its limited space. Each time failure is encountered, the pole-cart system is reset with an updated value function. The performance of the model over 207 failures showing the number of actions the model took before failure occured is plotted below. 

![](https://github.com/shenrunzhang/ml_solutions_cs229/blob/main/problemset_solutions/code/cartpole_qlearning/control.pdf)


## Weighted Logisitic Regression

