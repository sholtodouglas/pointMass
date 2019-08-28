from gym.envs.registration import register

register(
    id='pointMass-v0',
    entry_point='pointMass.envs:pointMassEnv',
)

register(
    id='pointMassObject-v0',
    entry_point='pointMass.envs:pointMassEnvObject',
)

register(
    id='pointMassObjectDense-v0',
    entry_point='pointMass.envs:pointMassEnvObjectDense',
)