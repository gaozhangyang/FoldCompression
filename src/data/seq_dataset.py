from nemo.utils import logging

def build_train_dataset(
    cluster_file,
    db_path,
    max_train_steps,
    global_batch_size,
    seed,
    max_seq_length,
    mask_prob,
    mask_token_prob,
    mask_random_prob,
    random_mask_strategy,
    tokenizer,
    dataset,
):
    num_train_samples = int(max_train_steps * global_batch_size)
    return dataset.create_train_dataset(
        cluster_file=cluster_file,
        db_path=db_path,
        total_samples=num_train_samples,
        seed=seed,
        max_seq_length=max_seq_length,
        mask_prob=mask_prob,
        mask_token_prob=mask_token_prob,
        mask_random_prob=mask_random_prob,
        random_mask_strategy=random_mask_strategy,
        tokenizer=tokenizer,
    )


def build_valid_dataset(
    cluster_path,
    db_path,
    seed,
    max_seq_length,
    mask_prob,
    mask_token_prob,
    mask_random_prob,
    random_mask_strategy,
    tokenizer,
    dataset,
    trainer,
    data_sampler,
    infer_num_samples_fn,
):
    val_clusters = dataset.create_valid_clusters(cluster_path)
    
    if trainer.limit_val_batches == 0:
        logging.info("Skip creating validation dataset because trainer.limit_val_batches=0.")
        return None

    num_val_samples = infer_num_samples_fn(
        limit_batches=trainer.limit_val_batches,
        num_samples_in_dataset=len(val_clusters),
        global_batch_size=data_sampler.global_batch_size,
        stage="val",
    )

    return dataset.create_valid_dataset(
        clusters=cluster_path,
        db_path=db_path,
        total_samples=num_val_samples,
        seed=seed,
        max_seq_length=max_seq_length,
        mask_prob=mask_prob,
        mask_token_prob=mask_token_prob,
        mask_random_prob=mask_random_prob,
        random_mask_strategy=random_mask_strategy,
        tokenizer=tokenizer,
    )
