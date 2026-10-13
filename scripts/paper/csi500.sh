export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name=TimeXer_paper



python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_patch0_gru \
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
  --patch_len 0

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_patch4_gru \
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
  --patch_len 4

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_patch8_gru \
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
  --patch_len 8

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_patch32_gru \
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
  --patch_len 32


python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_knn8_gru \
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
  --knn 8 \
  --embed_dim 16 \
  --use_multi_gpu \
  --stocknums 310

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_knn32_gru \
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
  --knn 32 \
  --embed_dim 16 \
  --use_multi_gpu \
  --stocknums 310

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_knn64_gru \
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
  --knn 64 \
  --embed_dim 16 \
  --use_multi_gpu \
  --stocknums 310

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_nods_noct_graph_gru \
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
  --ct_flag \
  --ds_flag

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_noct_graph_gru \
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
  --ct_flag

python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_nods_ct_graph_gru \
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
  --ds_flag


python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_ds_ct_graph_gru \
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
  --stocknums 310



python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
  --model_id csi500_nods_indusgeo_graph_gru \
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
# # python -u /disk4/cl/yc/yc/transformer/occupy_gpu.py



# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_patch0 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 16 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310 \
#   --patch_len 0

# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_patch4 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 16 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310 \
#   --patch_len 4

# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_patch8 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 16 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310 \
#   --patch_len 8

# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_patch32 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 16 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310 \
#   --patch_len 32



# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_knn8 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 8 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310

# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_knn32 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 32 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310

# python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
#   --model_id csi500_ds_ct_graph_knn64 \
#   --model $model_name \
#   --data stock3geo \
#   --features MS \
#   --seq_len 64 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 114 \
#   --dec_in 114 \
#   --c_out 114 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 240 \
#   --learning_rate 3e-5  \
#   --batch_size 2 \
#   --knn 64 \
#   --embed_dim 16 \
#   --use_multi_gpu \
#   --stocknums 310
# # python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
# #   --model_id csi500_nods_noct_graph \
# #   --model $model_name \
# #   --data stock3geo \
# #   --features MS \
# #   --seq_len 64 \
# #   --label_len 0 \
# #   --pred_len 24 \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 114 \
# #   --dec_in 114 \
# #   --c_out 114 \
# #   --des 'Exp' \
# #   --itr 1 \
# #   --train_epochs 240 \
# #   --learning_rate 3e-5  \
# #   --batch_size 2 \
# #   --knn 16 \
# #   --embed_dim 16 \
# #   --use_multi_gpu \
# #   --stocknums 310 \
# #   --ds_flag \
# #   --ct_flag


# # python -u /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/run.py \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path /disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_csi500 \
# #   --model_id csi500_nods_noct_nograph \
# #   --model $model_name \
# #   --data stock3geo \
# #   --features MS \
# #   --seq_len 64 \
# #   --label_len 0 \
# #   --pred_len 24 \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 114 \
# #   --dec_in 114 \
# #   --c_out 114 \
# #   --des 'Exp' \
# #   --itr 1 \
# #   --train_epochs 240 \
# #   --learning_rate 3e-5  \
# #   --batch_size 2 \
# #   --knn 16 \
# #   --embed_dim 16 \
# #   --use_multi_gpu \
# #   --stocknums 310 \
# #   --graph_flag \
# #   --ds_flag \
# #   --ct_flag


# # python -u /disk4/cl/yc/yc/transformer/occupy_gpu.py


