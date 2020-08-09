from gym.envs.registration import registry, register, make, spec

register(

    id='grid4d-v0',

    entry_point='gym.envs:grid4d',
)
