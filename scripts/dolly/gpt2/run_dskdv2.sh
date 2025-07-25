#! /bin/bash
GPUS=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=${1-$(IFS=,; echo "${GPUS[*]}")}

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=/path/to/DSKD_v2
CKPT_TYPE="gpt2"
CKPT_NAME="gpt2-base"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"
TEACHER_MODEL_TYPE="gpt2"
TEACHER_MODEL_NAME="gpt2-xlarge"
TEACHER_MODEL_PATH="/path/to/teacher/model"
# data
DATA_DIR="${BASE_PATH}/data/dolly/"
# task
TASK="dskd_v2"
# hp
BATCH_SIZE=8
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=32
EPOCH=20
KD_RATE=0.5
KD_TEMP=2.0
# distiller
PROJECTOR_LR=0.0005
TOPK_VOCAB=-1
# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
CRITERION="dual_space_kd_v2"
KD_OBJ="forward_kl"   # [forward_kl, reverse_kl, js_divergence, skewed_forward_kl, skewed_reverse_kl, adaptive_kl]
CONFIG="${KD_OBJ}-${PRECISION}"
SETTING="${CONFIG}__teacher_${TEACHER_MODEL_NAME}__kd^rate${KD_RATE}__kd^temp${KD_TEMP}__epoch${EPOCH}__bsz${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}x${NNODES}_$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr${LR}__proj^lr${PROJECTOR_LR}"
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=3
# seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --model-dtype ${PRECISION}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# task
OPTS+=" --task ${TASK}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
# distiller
OPTS+=" --init-t2s-projector"
OPTS+=" --init-s2t-projector"
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --topk-vocab ${TOPK_VOCAB}"
# OPTS+=" --projector-path ${PROJECTOR_PATH}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
# OPTS+=" --on-policy"
# OPTS+=" --on-policy-after-n-epochs 1"
# OPTS+=" --stu-gen-ratio 1.0"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"

# ${CMD}
${CMD} \
2>&1 | tee -a ${SAVE_PATH}/train.log
