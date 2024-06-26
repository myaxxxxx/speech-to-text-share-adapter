tgt=$1
exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.st_adaptive
fairseq-train data/mustc/en-$tgt --text-data data/mustc/en-$tgt/binary/ --tgt-lang $tgt \
  --user-dir st \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 2000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation_with_oracle_reg_adaptive --reg-loss-type jsd --reg-weight 1.0 --label-smoothing 0.1 \
  --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --use-word-level-oracle --decay-k 15 --use-word-gumbel-noise --gumbel-temperature 1.0 \
  --adaptive-base 0.7 --adaptive-scale 0.05 --adaptive-func linear_cosine --adaptive-weight-drop 0.0 \
  --adaptive-st-loss --adaptive-mt-loss --adaptive-reg-loss \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --layernorm-embedding \
  --patience 10 \
  --fp16 \
  --st-training \
  --hubert-model-path checkpoints/hubert_base_ls960.pt \
  --eval-bleu \
  --eval-bleu-args '{"beam": 8}' \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --restore-file checkpoints/en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.st/checkpoint20.pt