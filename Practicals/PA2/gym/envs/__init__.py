from gym.envs.registration import registry, register, make, spec

register(

    id='Grid-v0',

    entry_point='gym.envs:GridWorld',
)
register(
    id='chakra-v0',
    entry_point='gym.envs:chakra',
)
register(
    id='vishamC-v0',
    entry_point='gym.envs:vishamC',
)