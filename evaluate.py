from d3rlpy.algos import CQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from lhd_env_nodes.tasks.reach_collision import LHDReachCollision


env = LHDReachCollision(max_episode_steps=250,
                        success_reward=750,
                        collision_penalization=-1000,
                        dense_reward_weight=10,
                        publish_collision_points=False)

scorer = evaluate_on_environment(env)
cql = CQL()
cql.build_with_env(env)
mean_episode_return = scorer(cql)
