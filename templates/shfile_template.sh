#!/bin/bash
TEMPLATE_NAME=template-diffusion

# Create only the first time
# oc create -f ${TEMPLATE_NAME}

echo "Submitting job..."
oc process ${TEMPLATE_NAME} \
    -p JOBNAME=diffusion-test \
    -p CONTAINERNAME=diffusion-container \
    -p IMAGE=marcoaversa/ddpm:v0 \
    -p GPUREQ=1 \
    -p GPULIM=1 \
    -p COMMAND=python3 \
    -p ARG1=/nfs/conditionalDDPM/train.py \
    -p ARG2=--dataset=MNIST \
    -p ARG3=--dataset_path=/nfs/conditionalDDPM/data \
    -p ARG4=--logdir=/nfs/conditionalDDPM/logs \
    -p ARG5=--save_sample_every=1000 \
    -p ARG6=--mode=train \
    -p ARG7=--image_size=28 \
    -p ARG8=--timesteps=1000 \
    -p ARG9=--loss=l1 \
    -p ARG10=--lr=0.00002 \
    -p ARG11=--batch_size=32 \
    -p GPUTYPE=gputitan \
    | oc create -f -
   
