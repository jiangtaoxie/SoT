NODE_NUM=8
DATA_ROOT=/path/to/data
MODEL_NAME=SoT_Base
BATCH_SIZE=64
INIT_LR=5e-4
WEIGHT_DECAY=.065
RESOLUTION=224


python3 -m torch.distributed.launch --nproc_per_node=$NODE_NUM main.py \
$DATA_ROOT \
--model $MODEL_NAME -b $BATCH_SIZE --lr  $INIT_LR \
--weight-decay $WEIGHT_DECAY \
--img-size $RESOLUTION \
--amp