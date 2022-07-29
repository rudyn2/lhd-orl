from argparse import Action
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
from d3rlpy.models.encoders import VectorEncoderFactory
import argparse



def main(config: dict):
    with open(os.getenv("RB_PATH"), "rb") as f:
        sb3_buffer = pickle.load(f, fix_imports=True)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=sb3_buffer.observations.squeeze(1),
        actions=sb3_buffer.actions.squeeze(1),
        rewards=sb3_buffer.rewards.squeeze(1),
        terminals=sb3_buffer.dones.squeeze(1),
    )

    # here i should create my own encoders
    encoder_factory = VectorEncoderFactory(hidden_units=["768, 512"])

    # setup CQL algorithm
    cql = CQL(use_gpu=True, 
            actor_encoder_factory=encoder_factory, 
            critic_encoder_factory=encoder_factory)

    # split train and test episodes
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    # start training
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=100,
            scorers={
                'advantage': discounted_sum_of_advantage_scorer, # smaller is better
                'td_error': td_error_scorer, # smaller is better
                'value_scale': average_value_estimation_scorer # smaller is better
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple experiments")

    parser.add_argument("--checkpoint", type=str,
                        default="/home/rudy/lhd_gazebo_ws/src/lhd_navigation_ml/lhd_navigation_rl/src/actor_last.pth")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--conservative_weight", type=float, default=5.0)

    args = parser.parse_args()
    config = vars(args)
    main(config)