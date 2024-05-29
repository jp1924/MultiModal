export WANDB_PROJECT="UReader"
export WANDB_RUN_GROUP="pretrain"

export WANDB_WATCH="none"
export WANDB_DISABLED="false"
export WANDB_DISABLE_CODE="false"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export OMP_NUM_THREADS=2

deepspeed --num_gpus=4 \
    /root/workspace/mplug_owl_pretrain.py \
    --output_dir=/root/output_dir/pretrain \
    --run_name=mPlug_Owl-pretrain \
    --vision_model_name_or_path=Bingsu/clip-vit-large-patch14-ko \
    --language_model_name_or_path=maywell/Synatra-7B-v0.3-dpo \
    --preprocessing_num_workers=20 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_cache=false \
    --cache_dir=false \
    --num_train_epochs=2 \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --report_to=wandb \
    --learning_rate=1e-4 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.02 \
    --weight_decay=0.05 \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --eval_steps=10000 \
    --save_steps=10000 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --dataset_names jp1924/KoreanImageCaptioningDataset jp1924/OutsideKnowledgebasedMultimodalQAData jp1924/VisualQuestionAnswering \
    --train_dataset_prefix train \
    --valid_dataset_prefix validation \
    --test_dataset_prefix test \
    --num_query_tokens=32 \
    --vision_projection_bias=false \
    --abstractor_num_hidden_layers=6 \
    --abstractor_num_attention_heads=16 \
    --abstractor_intermediate_size=2048 \
    --abstractor_attention_probs_dropout_prob=0.01 \
    --abstractor_layer_norm_eps=1e-6 \
    --abstractor_encoder_hidden_size=1024 \
    --attn_implementation=flash_attn \
    --deepspeed=/root/workspace/ds_config/ZeRO_2.json