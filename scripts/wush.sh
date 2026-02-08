export OMP_NUM_THREADS=16
export HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"


MODEL=${MODEL:-"meta-llama/Llama-3.2-3B"}
MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
# Data params
NUM_SEQUENCES=${NUM_SEQUENCES:-1024}
SEED=${SEED:-0}

# Quantization params
FORMAT=${FORMAT:-"mxfp"}
W_BITS=${W_BITS:-4}
A_BITS=${A_BITS:-4}
W_GROUP_SIZE=${W_GROUP_SIZE:-32}
A_GROUP_SIZE=${A_GROUP_SIZE:-32}
GPTQ=${GPTQ:-0}
W_OBSERVER=${W_OBSERVER:-"minmax"}
# Save params
EXPORT_QUANTIZATION=${EXPORT_QUANTIZATION:-"realquant"}
# Transform params
TRANSFORM_CLASS=${TRANSFORM_CLASS:-"wush"}
HADAMARD_GROUP_SIZE=${HADAMARD_GROUP_SIZE:-32}
# Evaluation params
EVAL_PERPLEXITY=${EVAL_PERPLEXITY:-1}
EVAL_OPENLLM=${EVAL_OPENLLM:-1}
EVAL_PLATINUMBENCH=${EVAL_PLATINUMBENCH:-1}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-"auto"}
LM_EVAL_BATCH_SIZE_MMLU_GSM8K=${LM_EVAL_BATCH_SIZE_MMLU_GSM8K:-"auto"}
# Misc params
LOG_WANDB=${LOG_WANDB:-0}
DTYPE=${DTYPE:-"auto"}
CPU_OFFLOAD_ACTIVATIONS=${CPU_OFFLOAD_ACTIVATIONS:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"WUSH"}

SCRIPT_ARGS=""

if [[ $GPTQ == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"
fi

if [[ $EVAL_PERPLEXITY == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity"
fi

if [[ $EVAL_OPENLLM == 0 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_openllm"
fi
if [[ $EVAL_PLATINUMBENCH == 0 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_platinumbench"
fi

if [[ $LOG_WANDB == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --log_wandb"
fi

METHOD_NAME=""
if [[ $GPTQ == 1 ]]; then
    METHOD_NAME="GPTQ"
else
    METHOD_NAME="RTN"
fi

if [[ $CPU_OFFLOAD_MODULES == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_modules"
fi

if [[ $CPU_OFFLOAD_ACTIVATIONS == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_activations"
fi

export WANDB_PROJECT=$WANDB_PROJECT
export WANDB_NAME=${MODEL}/${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}

if [[ $EXPORT_QUANTIZATION == "realquant" || $EXPORT_QUANTIZATION == "pseudoquant" ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --export_quantized_model ${EXPORT_QUANTIZATION}"
    if [[ $EXPORT_QUANTIZATION == "realquant" ]]; then
        SAVE_DIR=quantized_models
    else
        SAVE_DIR=pseudoquantized_models
    fi
fi

echo "Starting quantization with the following settings:" 
echo "Model: $MODEL"
echo "Format: $FORMAT"
echo "Weight Bits: $W_BITS"
echo "Activation Bits: $A_BITS"
echo "Weight Group Size: $W_GROUP_SIZE"
echo "Activation Group Size: $A_GROUP_SIZE"
echo "Transform Class: $TRANSFORM_CLASS"
echo "Weight Observer: $W_OBSERVER"
echo "SEED : $SEED"
echo "Hadamard Group Size: $HADAMARD_GROUP_SIZE"
echo "LM Eval Batch Size: $LM_EVAL_BATCH_SIZE"
echo "LM Eval Batch Size MMLU/GSM8K: $LM_EVAL_BATCH_SIZE_MMLU_GSM8K"
echo "LM Eval Batch Size WANDB_PROJECT: $WANDB_PROJECT"

python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=${FORMAT} \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --w_group_size=${W_GROUP_SIZE} \
    --a_group_size=${A_GROUP_SIZE} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    $SCRIPT_ARGS \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=fineweb-edu \
    --num_sequences=${NUM_SEQUENCES} \
    --sequence_length=2048 \
    --dtype=${DTYPE} \
    --lm_eval_batch_size=${LM_EVAL_BATCH_SIZE} \
    --lm_eval_batch_size_mmlu_gsm8k=${LM_EVAL_BATCH_SIZE_MMLU_GSM8K} \
    --save_path "${SAVE_DIR}/${MODEL_ID}-${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}" \
    --cpu_offload_activations \
    --disable_thinking \
    --cpu_offload_modules \
    --seed=${SEED} \
    --fuse_global_scale \
    --amp