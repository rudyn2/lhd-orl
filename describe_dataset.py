from load_dataset import load_dataset
import sys
from pprint import pprint
from d3rlpy.dataset import MDPDataset


def describe(dataset: MDPDataset):
    stats = dataset.compute_stats()
    print(f"# Episodes: {len(dataset.episodes)}")
    print(f"# Transitions: {sum(len(e) for e in dataset.episodes)}")
    dataset.build_episodes()
    stats = dataset.compute_stats()
    dataset.dump('data/merged_1000_scaled.h5')
    pprint(stats['return'])


if __name__ == "__main__":
    dataset = load_dataset(sys.argv[1])
    describe(dataset)
