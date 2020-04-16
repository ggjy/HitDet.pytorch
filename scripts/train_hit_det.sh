#!/usr/bin/env bash

cd ../

GPUs=8
CONFIG='configs/nas_trinity/2stage_hitdet.py'
WORKDIR='./work_dirs/hitdet_1x/'

python -m torch.distributed.launch \
--nproc_per_node=${GPUs} train.py \
--validate \
--gpus ${GPUs} \
--launcher pytorch \
--config ${CONFIG} \
--work_dir ${WORKDIR}
