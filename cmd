CLI Log
-------
Debra_Davis: export EXP_PATH=`/workspace/scripts/gen_dir.sh /workspace/experiments/language_modeling` && PYTHONPATH=/workspace/language_modeling/src \
MODEL=GPT DATA=Pile nohup python runner.py fit --model.batch_size 8 --model.max_seq_len 1024 --model.num_layers 7 --model.num_heads 12 \
--model.d_model 3840 --model.dropout 0.1 --data.context_len 512 --data.tokenizer_path /workspace/data/pile/tokenizer/50000.model \
--data.path /workspace/data/pile --optimizer=AdamW --optimizer.lr .0001 --lr_scheduler=CosineWarmupScheduler --lr_scheduler.warmup 25000 \
--lr_scheduler.total_steps 4854349 --model.lr_scheduler_interval step --trainer.callbacks LearningRateMonitor --trainer.callbacks ModelCheckpoint \
--trainer.callbacks.save_top_k 3 --trainer.callbacks.monitor Validation/BPB --trainer.callbacks.dirpath $EXP_PATH/checkpoints --trainer.strategy ddp_sharded \
--trainer.gpus 4 --trainer.val_check_interval 50000 --trainer.max_time 30:00:00:00 --trainer.precision 16 --trainer.accumulate_grad_batches 4 &

Carolyn_Mayo: export EXP_PATH=`/workspace/scripts/gen_dir.sh /workspace/experiments/language_modeling` && PYTHONPATH=/workspace/language_modeling/src \
MODEL=GPT DATA=Pile nohup python runner.py fit --model.batch_size 32 --model.max_seq_len 1024 --model.num_layers 6 --model.num_heads 8 --model.d_model 1024 \
--model.dropout 0.1 --data.context_len 512 --data.tokenizer_path /workspace/data/pile/tokenizer/50000.model \
--data.path /workspace/data/pile --optimizer=AdamW --optimizer.lr .0001 --lr_scheduler=CosineWarmupScheduler \
--lr_scheduler.warmup 5000 --lr_scheduler.total_steps 5337894 --model.lr_scheduler_interval step --trainer.callbacks LearningRateMonitor \
--trainer.callbacks ModelCheckpoint --trainer.callbacks.save_top_k 3 --trainer.callbacks.monitor Validation/BPB \
--trainer.callbacks.dirpath $EXP_PATH/checkpoints --trainer.strategy ddp_sharded --trainer.gpus 4 --trainer.val_check_interval 50000 \
--trainer.max_time 10:00:00:00 &

Helen_Doughty: export EXP_PATH=`/workspace/scripts/gen_dir.sh /workspace/experiments/language_modeling` && PYTHONPATH=/workspace/language_modeling/src \
MODEL=GPT DATA=Pile nohup python runner.py fit --model.batch_size 32 --model.seq_len 1024 --model.num_layers 6 --model.num_heads 8 \
--model.d_model 1024 --model.dropout 0.1 --data.tokenizer_path /workspace/data/pile/tokenizer/50000.model --data.path /workspace/data/pile \
--optimizer=AdamW --optimizer.lr .0001 --lr_scheduler=CosineWarmupScheduler --lr_scheduler.warmup 5000 --lr_scheduler.total_steps 3366428 \
--model.lr_scheduler_interval step --trainer.callbacks LearningRateMonitor --trainer.callbacks ModelCheckpoint --trainer.callbacks.save_top_k 3 \
--trainer.callbacks.monitor Validation/BPB --trainer.strategy ddp_sharded --trainer.gpus 4 --trainer.val_check_interval 50000 \
--trainer.max_time 07:00:00:00 &

