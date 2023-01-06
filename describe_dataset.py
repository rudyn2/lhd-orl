from load_dataset import load_dataset
import sys
from pprint import pprint


if __name__ == "__main__":
    dataset = load_dataset(sys.argv[1])
    stats = dataset.compute_stats()
    pprint(stats['return'])
    returns = []
    for episode in dataset.episodes:
        if episode.rewards[-1] < -500:
            episode.rewards[-1] = -500 
    dataset.build_episodes()
    stats = dataset.compute_stats()
    dataset.dump('data/merged_1000_scaled.h5')
    pprint(stats['return'])

