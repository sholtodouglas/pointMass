from gym.envs.registration import register

register(
    id='pointMass-v0',
    entry_point='pointMass.envs:pointMassEnv',
)

register(
    id='pointMassDense-v0',
    entry_point='pointMass.envs:pointMassEnvDense',
)

register(
    id='pointMassObject-v0',
    entry_point='pointMass.envs:pointMassEnvObject',
)

register(
    id='pointMassObjectDuo-v0',
    entry_point='pointMass.envs:pointMassEnvObjectDuo',
)

register(
    id='pointMassObjectDense-v0',
    entry_point='pointMass.envs:pointMassEnvObjectDense',
)