




fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
  --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
  --save-dir ${MULTILINGUAL_BACKBONE} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_m --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${PRETRAINED_ASR}




export CUDA_VISIBLE_DEVICES=0

SAVE_DIR=/workspace/s2t/mustc/en-de/save_dir/multi

data_dir=/workspace/s2t/mustc/en-de
TEXT_DIR=/workspace/s2t/mustc/en-de/delta_data_bin
st_DIR=/workspace/s2t/st_orig
HU_BERT=/workspace/s2t/mustc/en-de/hu_bert


exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.multi_test
fairseq-train $data_dir --text-data $TEXT_DIR  \
  --user-dir $st_DIR \
  --config-yaml config.yaml --train-subset de_train, nl_train, es_train, fr_train, it_train, pt_train, ro_train, ru_train --valid-subset de_dev, nl_dev, es_dev, fr_dev, it_dev, pt_dev, ro_dev, ru_dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 3000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --patience 10 \
  --fp16 \
  --st-training --mt-finetune \
  --hubert-model-path $HU_BERT/hubert_base_ls960.pt \
  --max-source-positions 512 --max-target-positions 512



tgt=$1
exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt
fairseq-train data/mustc/en-$tgt --text-data data/mustc/en-$tgt/binary/ --tgt-lang $tgt \
  --user-dir st \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 2000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --layernorm-embedding \
  --patience 10 \
  --fp16 \
  --st-training --mt-finetune \
  --hubert-model-path checkpoints/hubert_base_ls960.pt \
  --eval-bleu \
  --eval-bleu-args '{"beam": 8}' \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --load-pretrained-mt-encoder-decoder-from checkpoints/en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain/avg_last_10_epoch.pt






ckpt=/workspace/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v2
python /workspace/s2t/fairseq/scripts/average_checkpoints.py \
    --inputs $ckpt \
    --num-epoch-checkpoints 10 \
    --output $ckpt/avg_last_10_epoch.pt




# # ckpt=/workspace/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_ft_v2/checkpoint1.pt
# ckpt=/workspace/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_mt_deltalm_test/checkpoint1.pt
# st_DIR=/workspace/s2t/st
# lang=de
# lenpen=1.2


# data_dir=/workspace/s2t/mustc/en-de
# TEXT_DIR=/workspace/s2t/mustc/en-de/delta_data_bin
# st_DIR=/workspace/s2t/st
# HU_BERT=/workspace/s2t/mustc/en-de/hu_bert

# export CUDA_VISIBLE_DEVICES=3
# fairseq-generate  $data_dir \
#   --user-dir $st_DIR \
#   --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
#   --path $ckpt \
#   --max-source-positions 900000 \
#   --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu

