ckpt=/workspace/s2t/en-de/checkpoints/en-de.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt/avg_last_10_epoch.pt
st_DIR=/workspace/s2t/st
lang=de
lenpen=1.2


data_dir=/workspace/s2t/mustc/en-de
TEXT_DIR=/workspace/s2t/mustc/en-de/delta_data_bin
st_DIR=/workspace/s2t/st
HU_BERT=/workspace/s2t/mustc/en-de/hu_bert

export CUDA_VISIBLE_DEVICES=3
fairseq-generate  $data_dir \
  --user-dir $st_DIR \
  --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
  --path $ckpt \
  --max-source-positions 900000 \
  --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu



# /home/SuXiangDong/SxdStu98/student/st/fairseq/checkpoints/avg_last_10_epoch.pt