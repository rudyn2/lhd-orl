import d3rlpy
from d3rlpy.datasets import get_cartpole
import numpy as np
import pickle5 as pickle    # noqa
import os
from d3rlpy.algos import CQL
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import evaluate_on_environment


# so far we have needed: pickle5, stable-baselines3, matplotlib

with open(os.getenv("RB_PATH"), "rb") as f:
    sb3_buffer = pickle.load(f, fix_imports=True)

dataset = d3rlpy.dataset.MDPDataset(
    observations=sb3_buffer.observations.squeeze(1),
    actions=sb3_buffer.actions.squeeze(1),
    rewards=sb3_buffer.rewards.squeeze(1),
    terminals=sb3_buffer.dones.squeeze(1),
)


# setup CQL algorithm
cql = CQL(use_gpu=True, reward_scaler="standard")

# split train and test episodes
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=2,
        scorers={
            'advantage': discounted_sum_of_advantage_scorer, # smaller is better
            'td_error': td_error_scorer, # smaller is better
            'value_scale': average_value_estimation_scorer # smaller is better
        })