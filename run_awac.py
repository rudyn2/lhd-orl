import d3rlpy
import pickle5 as pickle   
from d3rlpy.algos import AWAC
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer, \
    td_error_scorer, average_value_estimation_scorer, initial_state_value_estimation_scorer
from sklearn.model_selection import train_test_split
import argparse
import wandb
from wandb_callback import wandb_callback
    

def main(config: dict):
    with open(config['data'], "rb") as f:
        sb3_buffer = pickle.load(f, fix_imports=True)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=sb3_buffer.observations.squeeze(1),
        actions=sb3_buffer.actions.squeeze(1),
        rewards=sb3_buffer.rewards.squeeze(1),
        terminals=sb3_buffer.dones.squeeze(1),
    )
    print(f"Buffer loaded with size: {dataset.size()}")

    # setup algorithm
    awac = AWAC(reward_scaler="standard",
              actor_learning_rate=config["actor_lr"],
              critic_learning_rate=config["critic_lr"],
              actor_encoder_factory=config["encoder_type"],
              critic_encoder_factory=config["encoder_type"],
              batch_size=config["batch_size"],
              lam=config["lam"],
              n_action_samples=config["n_action_samples"],
              use_gpu=True)
    
    base_config = awac.get_params()
    base_config.update(config)
    base_config = {k: v for k,v in base_config.items() 
                   if isinstance(v, str) or isinstance(v, int) or isinstance(v, float)}

    # split train and test episodes
    train_episodes, test_episodes = train_test_split(dataset, test_size=config["test_size"])
    
    # init wandb
    wandb.init(project="lhd-orl",
               config=base_config,
               tags=['AWAC'])


    # # start training
    awac.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=config["n_epochs"],
            scorers={
                'advantage': discounted_sum_of_advantage_scorer, # smaller is better,
                'td_error': td_error_scorer, # smaller is better
                'value_scale': average_value_estimation_scorer, # smaller is better
                'initial_value': initial_state_value_estimation_scorer
            },
            callback=wandb_callback,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CRR experiments")
    
    common_params = parser.add_argument_group('common')
    algo_params = parser.add_argument_group('algo')
    
    # common parameters
    common_params.add_argument("--data", type=str)
    common_params.add_argument("--test_size", type=float, default=0.2)
    common_params.add_argument("--n_epochs", type=int, default=10)
    common_params.add_argument("--encoder_type", type=str, default="default")
    
    # algo parameters        
    algo_params.add_argument("--actor_lr", type=float, default=3e-4)
    algo_params.add_argument("--critic_lr", type=float, default=3e-4)
    algo_params.add_argument("--batch_size", type=int, default=1024)
    algo_params.add_argument("--lam", type=float, default=1.0)
    algo_params.add_argument("--n_action_samples", type=int, default=1)

    args = parser.parse_args()
    config = vars(args)
    main(config)