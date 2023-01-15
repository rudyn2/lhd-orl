#!/bin/bash
data_path="data/teleop_1000.h5"
python run_iql.py --data $data_path --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_cql.py --data $data_path --alpha_threshold=10 --conservative_weight=1 --encoder_type=dense --soft_q_backup=False --batch_size 256 --n_epochs 50
python run_crr.py --data $data_path --actor_lr=3e-05 --advantage_type=mean --critic_lr=0.0003 --encoder_type=dense --n_critics=1 --weight_type=exp --n_epochs=50 --batch_size=256
python run_awac.py --data $data_path --encoder_type=dense --lam=1 --n_action_samples=1 --batch_size=256 --n_epochs=50