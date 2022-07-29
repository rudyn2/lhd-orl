import d3rlpy
import pickle5 as pickle    # noqa
import os
import numpy as np
import csv



def export_vector_observation_dataset_as_csv(dataset, fname):
    observation_size = dataset.get_observation_shape()[0]
    action_size = dataset.get_action_size()

    with open(fname, "w") as file:
        writer = csv.writer(file)

        # write header
        header = ["episode"]
        header += ["observation:%d" % i for i in range(observation_size)]
        if dataset.is_action_discrete():
            header += ["action:0"]
        else:
            header += ["action:%d" % i for i in range(action_size)]
        header += ["reward"]
        writer.writerow(header)

        for i, episode in enumerate(dataset.episodes):
            # prepare data to write
            observations = np.asarray(episode.observations)
            episode_length = observations.shape[0]
            actions = np.asarray(episode.actions).reshape(episode_length, -1)
            rewards = episode.rewards.reshape(episode_length, 1)
            episode_ids = np.full([episode_length, 1], i)

            # write episode
            rows = np.hstack([episode_ids, observations, actions, rewards])
            writer.writerows(rows)

with open("data/rb_1000_20220727T2221.pkl", "rb") as f:
    sb3_buffer = pickle.load(f, fix_imports=True)

dataset = d3rlpy.dataset.MDPDataset(
    observations=sb3_buffer.observations.squeeze(1),
    actions=sb3_buffer.actions.squeeze(1),
    rewards=sb3_buffer.rewards.squeeze(1),
    terminals=sb3_buffer.dones.squeeze(1),
)

export_vector_observation_dataset_as_csv(dataset, fname="datasetv1.csv")