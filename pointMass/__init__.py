from gym.envs.registration import register

register(
    id='pointMass-v0',
    entry_point='pointMass.envs:pointMassEnv',
)