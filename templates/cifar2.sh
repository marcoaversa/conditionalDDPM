#!/bin/bash
TEMPLATE_NAME=template-diffusion

# Create only the first time
# oc create -f ${TEMPLATE_NAME}

echo "Submitting job..."
oc process ${TEMPLATE_NAME} \
    -p JOBNAME=diff-cifar-2exp \
    -p CONTAINERNAME=diffusion-container \
    -p IMAGE=marcoaversa/ddpm:v0 \
    -p GPUREQ=1 \
    -p GPULIM=1 \
    -p COMMAND=python3 \
    -p ARG1=/nfs/conditionalDDPM/train.py \
    -p ARG2=--dataset=CIFAR10 \
    -p ARG3=--dataset_path=/nfs/conditionalDDPM/data \
    -p ARG4=--logdir=/nfs/conditionalDDPM/logs/cifar2 \
    -p ARG5=--save_sample_every=4000 \
    -p ARG6=--mode=train \
    -p ARG7=--image_size=32 \
    -p ARG8=--timesteps=4000 \
    -p ARG9=--loss=l1 \
    -p ARG10=--lr=0.00002 \
    -p ARG11=--batch_size=32 \
    -p ARG12=--train_num_steps=400000 \
    -p GPUTYPE=gputitan \
    | oc create -f -
   
