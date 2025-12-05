# export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com/
lm_eval --model hf \
    --model_args pretrained=/data/llm/longchen/OPT-SNN/opt-13B/setthres,parallelize=True \
    --tasks wsc273 \
    --batch_size 1