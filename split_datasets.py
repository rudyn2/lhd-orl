import d3rlpy
import pickle5 as pickle   
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset


if __name__ == "__main__":
    data_path = "data/rb_1000_20221118T0816.pkl"
    with open(data_path, "rb") as f:
        sb3_buffer = pickle.load(f, fix_imports=True)

        dataset = d3rlpy.dataset.MDPDataset(
            observations=sb3_buffer.observations.squeeze(1),
            actions=sb3_buffer.actions.squeeze(1),
            rewards=sb3_buffer.rewards.squeeze(1),
            terminals=sb3_buffer.dones.squeeze(1),
        )

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    print()