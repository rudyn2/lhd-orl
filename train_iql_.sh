#!/bin/bash
replay_buffer_400k_1m="data/rb_1000_20221118T0816.pkl"
replay_buffer_400k_500k_good="data/rb_500_20221119T1503.pkl"
replay_buffer_200k_500k="data/rb_500_20221126T1526.pkl"
python run_iql.py --data $replay_buffer_400k_1m --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_iql.py --data $replay_buffer_400k_500k_good --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_iql.py --data $replay_buffer_200k_500k --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50