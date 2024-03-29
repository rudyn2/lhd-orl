from d3rlpy.algos import IQL
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer, \
    td_error_scorer, average_value_estimation_scorer, initial_state_value_estimation_scorer
from sklearn.model_selection import train_test_split
import argparse
import wandb
from wandb_callback import wandb_callback
from load_dataset import load_dataset


def main(config: dict):
    dataset = load_dataset(config["data"])
    print(f"Buffer loaded with size: {dataset.size()}")

    # setup algorithm
    iql = IQL(reward_scaler="standard",
              actor_learning_rate=config["actor_lr"],
              critic_learning_rate=config["critic_lr"],
              actor_encoder_factory=config["encoder_type"],
              critic_encoder_factory=config["encoder_type"],
              batch_size=config["batch_size"],
              n_critics=config["n_critics"],
              expectile=config["expectile"],
              weight_temp=config["weight_temp"],
              max_weight=config["max_weight"],
              use_gpu=True)

    base_config = iql.get_params()
    base_config.update(config)
    base_config = {k: v for k, v in base_config.items()
                   if isinstance(v, str) or isinstance(v, int) or isinstance(v, float)}

    # split train and test episodes
    train_episodes, test_episodes = train_test_split(dataset, test_size=config['test_size'])
    print(f"Train episodes: {len(train_episodes)}")
    print(f"Test episodes: {len(test_episodes)}")

    # init wandb
    wandb.init(project="lhd-orl",
               config=base_config,
               tags=['IQL'])

    # # start training
    iql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=config["n_epochs"],
            scorers={
                'advantage': discounted_sum_of_advantage_scorer,  # smaller is better,
                'td_error': td_error_scorer,  # smaller is better
                'value_scale': average_value_estimation_scorer,  # smaller is better,
                'initial_value': initial_state_value_estimation_scorer
            },
            callback=wandb_callback,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run iql experiments")

    common_params = parser.add_argument_group('common')
    algo_params = parser.add_argument_group('algo')

    # common parameters
    common_params.add_argument("--data", type=str)
    common_params.add_argument("--test_size", type=float, default=0.1)
    common_params.add_argument("--n_epochs", type=int, default=10)
    common_params.add_argument("--encoder_type", type=str, default="dense")

    # algo parameters
    algo_params.add_argument("--actor_lr", type=float, default=3e-4)
    algo_params.add_argument("--critic_lr", type=float, default=3e-4)
    algo_params.add_argument("--batch_size", type=int, default=256)
    algo_params.add_argument("--expectile", type=float, default=0.9)
    algo_params.add_argument("--weight_temp", type=float, default=10.0)
    algo_params.add_argument("--max_weight", type=float, default=50.0)
    algo_params.add_argument("--n_critics", type=int, default=2)

    args = parser.parse_args()
    config = vars(args)
    main(config)
