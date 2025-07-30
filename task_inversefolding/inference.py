import torch
from task_inversefolding.model_interface import BionemoLightningModule
from main import get_parser
from task_inversefolding.data_interface import ESMDataModule
from nemo import lightning as nl
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections import llm

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)  # Adjust key as per your dataset
            emb = model.extract_embedding(inputs)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def main():
    args = get_parser()
    args.data_splits = '1, 98, 1'  # Use the same splits as training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_module = ESMDataModule(
        cluster_path=args.cluster_path,
        database_path=args.database_path,
        global_batch_size=args.micro_batch_size,
        micro_batch_size=args.micro_batch_size,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_dataset_workers,
        random_mask_strategy=args.random_mask_strategy,
        prefix_len=args.prefix_len,
        data_splits=args.data_splits
    )
    data_module.setup()
    
    model = BionemoLightningModule(
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        hidden_dim=args.hidden_dim,
        prefix_len=args.prefix_len,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        scheduler_num_steps=args.scheduler_num_steps,
        custom_checkpoint_path = args.custom_checkpoint_path
    )
    model.configure_model()
    
    devices=args.num_gpus
    num_steps=args.num_steps
    limit_val_batches=args.limit_val_batches
    val_check_interval=args.val_check_interval
    log_every_n_steps=args.log_every_n_steps
    num_nodes=args.num_nodes
    precision=args.precision
    grad_reduce_in_fp32=args.grad_reduce_in_fp32
    decoder_first_pipeline_num_layers=args.decoder_first_pipeline_num_layers
    pipeline_model_parallel_size=args.pipeline_model_parallel_size
    tensor_model_parallel_size=args.tensor_model_parallel_size
    overlap_grad_reduce=not args.no_overlap_grad_reduce
    overlap_param_gather=not args.no_overlap_param_gather
    average_in_collective=not args.no_average_in_collective
    
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=get_autocast_dtype(precision),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=True,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        num_layers_in_first_pipeline_stage=decoder_first_pipeline_num_layers,
        ckpt_load_optimizer=False
    )
        
    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
    )
    
    for batch in data_module.train_dataloader():
        pass

    
    llm.validate(
        model=model,
        data=data_module,
        trainer=trainer
    )
    

if __name__ == "__main__":
    main()