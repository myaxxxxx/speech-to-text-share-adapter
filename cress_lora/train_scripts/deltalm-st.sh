target=ro
cd /workspace/s2t/raw_data/data/mustc/en-$target
export CUDA_VISIBLE_DEVICES=2
target=ro

SAVE_DIR=/workspace/s2t/deltalm_data/save_dir/$target/st_deltalm_adapters
pretrain_checkpoints_num=10

data_dir=/workspace/s2t/raw_data/data/mustc/en-$target
TEXT_DIR=/workspace/s2t/deltalm_data/en-$target/binary
CRESS_DIR=/workspace/s2t/cress_adapter
HU_BERT=/workspace/s2t/mustc/en-de/hu_bert


fairseq-train $data_dir --text-data $TEXT_DIR --tgt-lang $target \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir $SAVE_DIR --num-workers 4 --max-tokens 3000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch deltalm_transformer_adapter --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --patience 10 \
  --fp16 \
  --st-training --mt-finetune \
  --hubert-model-path $HU_BERT/hubert_base_ls960.pt \
  --pretrained_deltalm_checkpoint /workspace/s2t/deltalm_data/save_dir/$target/checkpoints/checkpoint$pretrain_checkpoints_num.pt \
  --max-source-positions 512 --max-target-positions 512 --adapters-bottle 128 



target=ro
cd /workspace/s2t/raw_data/data/mustc/en-$target

export CUDA_VISIBLE_DEVICES=1
ckpt=/workspace/s2t/deltalm_data/save_dir/$target/st_deltalm_adapters/checkpoint6.pt

tgt=ro


data_dir=/workspace/s2t/raw_data/data/mustc/en-$target
TEXT_DIR=/workspace/s2t/deltalm_data/en-$target/binary
CRESS_DIR=/workspace/s2t/cress_adapter
HU_BERT=/workspace/s2t/mustc/en-de/hu_bert

lenpen=0.8
export CUDA_VISIBLE_DEVICES=0
fairseq-generate  $data_dir \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
  --path $ckpt \
  --max-source-positions 900000 \
  --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu





target=ro
cd /workspace/s2t/raw_data/data/mustc/en-$target
export CUDA_VISIBLE_DEVICES=2
target=ro

SAVE_DIR=/workspace/s2t/deltalm_data/save_dir/$target/st_deltalm_adapters_test2
pretrain_checkpoints_num=10

data_dir=/workspace/s2t/raw_data/data/mustc/en-$target
TEXT_DIR=/workspace/s2t/deltalm_data/en-$target/binary
CRESS_DIR=/workspace/s2t/cress_adapter
HU_BERT=/workspace/s2t/mustc/en-de/hu_bert


fairseq-train $data_dir --text-data $TEXT_DIR --tgt-lang $target \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir $SAVE_DIR --num-workers 4 --max-tokens 3000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch deltalm_transformer_adapter --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
  --patience 10 \
  --fp16 \
  --st-training --mt-finetune \
  --hubert-model-path $HU_BERT/hubert_base_ls960.pt \
  --pretrained_deltalm_checkpoint /workspace/s2t/deltalm_data/save_dir/$target/checkpoints/checkpoint$pretrain_checkpoints_num.pt \
  --max-source-positions 512 --max-target-positions 512 --adapters-bottle 128 





export CUDA_VISIBLE_DEVICES=1
ckpt=/workspace/s2t/deltalm_data/save_dir/$target/st_deltalm_adapters_test2/checkpoint3.pt

tgt=ro


data_dir=/workspace/s2t/raw_data/data/mustc/en-$target
TEXT_DIR=/workspace/s2t/deltalm_data/en-$target/binary
CRESS_DIR=/workspace/s2t/cress_adapter
HU_BERT=/workspace/s2t/mustc/en-de/hu_bert

lenpen=0.8
fairseq-generate  $data_dir \
  --user-dir $CRESS_DIR \
  --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
  --path $ckpt \
  --max-source-positions 900000 \
  --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu