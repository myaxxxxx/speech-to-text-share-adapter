ckpt=xxx
python /workspace/s2t/fairseq/scripts/average_checkpoints.py \
    --inputs $ckpt \
    --num-epoch-checkpoints 10 \
    --output $ckpt/avg_last_10_epoch.pt