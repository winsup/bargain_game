from gym.envs.registration import register

register(
    id='one-rl-agent-v0',
    entry_point='gym_bargain.envs:OneRLAgentEnv',
)
register(
    id='two-rl-agent-v0',
    entry_point='gym_bargain.envs:TwoRLAgentEnv',
)