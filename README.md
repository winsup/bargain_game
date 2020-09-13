# The Dynamics of Bargaining Problem: A Reinforcement Learning Approach

This mini project attempts to analyze the bargaining problem with Deep Reinforcement Learning.

Install the Bargaining Game Environment by

    pip install -e gym-bargain

Import the Bargaining Game Environment by

    import gym
    env1 = gym.make('gym_bargain:one-rl-agent-v0')
    env2 = gym.make('gym_bargain:two-rl-agent-v0')

To customize the utility function or other parameters, 
users could clone this repository and alter the code in [gym-bargain folder](https://github.com/winsup/bargain_game/tree/master/gym-bargain)\
Jupyter Notebook Example is in [bargain_game_rl.ipynb](https://github.com/winsup/bargain_game/blob/master/bargain_game_rl.ipynb)
