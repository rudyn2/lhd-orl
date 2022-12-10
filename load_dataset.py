import d3rlpy
import pickle


def load_dataset(path: str) -> d3rlpy.dataset.MDPDataset:
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            sb3_buffer = pickle.load(f, fix_imports=True)

            dataset = d3rlpy.dataset.MDPDataset(
                observations=sb3_buffer.observations.squeeze(1),
                actions=sb3_buffer.actions.squeeze(1),
                rewards=sb3_buffer.rewards.squeeze(1),
                terminals=sb3_buffer.dones.squeeze(1),
            )
    elif path.endswith(".h5"):
        dataset = d3rlpy.dataset.MDPDataset.load(path)
    else:
        raise ValueError("Not supported dataset format.")

    return dataset
