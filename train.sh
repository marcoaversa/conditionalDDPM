#!/bin/sh

experiment_name='conditionalDDPM'
# experiment_name='dev'

#Dataset
image_size=256

#Model
timesteps=4000
dim=128
n_layers=4
model_type='c'
loss='l2'

#Hyperparameters
train_num_steps=1000
lr=0.00001
batch_size=8

for i in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
do
    echo "Running for step "$i
    run_name='ls_step'$i
    dataset='ls_step'$i
    python train.py --dataset=$dataset \
                --experiment_name=$experiment_name \
                --run_name=$run_name \
                --image_size=$image_size \
                --timesteps=$timesteps \
                --dim=$dim \
                --n_layers=$n_layers \
                --model_type=$model_type \
                --loss=$loss \
                --train_num_steps=$train_num_steps \
                --lr=$lr \
                --batch_size=$batch_size 
done