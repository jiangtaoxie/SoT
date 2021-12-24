NODE_NUM=8
DATA_ROOT=/path/to/data
MODEL_NAME=SoT_Tiny
BATCH_SIZE=128
INIT_LR=1e-3
WEIGHT_DECAY=.03
RESOLUTION=224


python3 -m torch.distributed.launch --nproc_per_node=$NODE_NUM main.py \
$DATA_ROOT \
--model $MODEL_NAME -b $BATCH_SIZE --lr  $INIT_LR \
--weight-decay $WEIGHT_DECAY \
--img-size $RESOLUTION \
--amp