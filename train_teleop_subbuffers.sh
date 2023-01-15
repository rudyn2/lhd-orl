#!/bin/bash
buffer_1="data/teleop_subbuffer_A.pkl"
buffer_2="data/teleop_subbuffer_B.pkl"
buffer_3="data/teleop_subbuffer_C.pkl"
buffer_4="data/teleop_subbuffer_D.pkl"
python run_iql.py --data $buffer_1 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_iql.py --data $buffer_2 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_iql.py --data $buffer_3 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_iql.py --data $buffer_4 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50