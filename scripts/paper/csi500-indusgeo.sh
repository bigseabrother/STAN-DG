export CUDA_VISIBLE_DEVICES=3,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name=TimeXer_paper

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_nods_indusgeo_graph \
  --model $model_name \
  --data stock3geo \
  --features MS \
  --seq_len 64 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 114 \
  --dec_in 114 \
  --c_out 114 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 240 \
  --learning_rate 3e-5  \
  --batch_size 2 \
  --knn 16 \
  --embed_dim 16 \
  --use_multi_gpu \
  --stocknums 310 \
  --index_num 28 \
  --ds_flag