export WANDB_PROJECT="UReader"
export WANDB_RUN_GROUP="pretrain"

export WANDB_WATCH="none"
export WANDB_DISABLED="true"
export WANDB_DISABLE_CODE="false"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export CUDA_VISIBLE_DEVICES="7"

export OMP_NUM_THREADS=2

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=61000 \
    /root/workspace/mplug_owl_pretrain.py \
    --output_dir=/root/output_dir/mplug_owl \
    --run_name=test \
    --cache_dir=/root/.cache/.mplug_owl_preprocessor \
    --model_name_or_path=/root/mplug_owl_init_model \
    --cache_file_name=preprocessor.arrow \
    --preprocessing_num_workers=6 \
    --preprocessing_batch_size=1000 \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=2 \
    --overwrite_cache=false \
    --num_train_epochs=3 \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --report_to=none \
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
    --dataset_names \
        jp1924/KoreanVisionDataforImageDescriptionSentenceExtractionandGeneration \
        jp1924/VisualQuestionAnswering \
        jp1924/KoreanImageCaptioningDataset \
        jp1924/OutsideKnowledgebasedMultimodalQAData \
    --train_dataset_prefix=train \
    --valid_dataset_prefix=validation \
    --test_dataset_prefix=test \
    --group_by_length=true \
    --deepspeed=/root/workspace/ds_config/ZeRO_2.json