#!/bin/bash
# Sample command to start a training job on borg using 2x2.
# Run from google3$ .
#
# Note: the command for running locally (with batch size 2) is:
#
# blaze-bin/experimental/users/vikrama/trainer \
# --config_file=experimental/users/vikrama/configs/transformer_expression_prediction_8gb.gin \
# --config=vikram_inputs.batch_size=2 \
# --alsologtostderr
#
# 25 freebie, 115 batch, 200 prod -- freebie use for now in Calico -- keep checkpointing somewhat regularly to not lose progress if kicked off
# jf = jellyfish, df = dragonfish
# dg donuts are 4x2 not 2x2
/google/data/ro/teams/dmgi/google_xmanager.par launch third_party/py/trax/google/xmanager/trainer.py -- \
  --xm_skip_launch_confirmation \
  --xm_resource_pool=brain \
  --xm_resource_alloc=user:vikrama \
  --cell=tm --tpu_type=jf --tpu_topology=2x2 --priority=25 --nomldash \
  --config_file=experimental/users/vikrama/configs/transformer_expression_prediction_8gb.gin \
  --job_name=CentralCoreDenseLayer2 \
  --trainer_build_target=//experimental/users/vikrama:trainer #\
#  --sweep_file experimental/users/vikrama/configs/sweep.yaml
