from load_dataset import load_dataset
import sys
from describe_dataset import describe


if __name__ == "__main__":
    dataset = load_dataset(sys.argv[1])

    print("BEFORE")
    describe(dataset)

    # Fix rewards
    for episode in dataset.episodes:
        if episode.rewards[-1] < -500:
            episode.rewards[-1] = -500
    dataset.build_episodes()

    print("AFTER")
    describe(dataset)
    dataset.dump(sys.argv[1])
