import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, create_optimizer
import tensorflow as tf
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args, _ = parser.parse_known_args()

    print("Read Training Data")
    dataset = load_dataset('kmfoda/booksum')
    
    model_checkpoint = 'google/pegasus-xsum'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="tf")
    
    def process(example):
        model_data = tokenizer(example['chapter'], truncation=True)
        model_label = tokenizer(example['summary'], truncation=True)
        model_data['labels'] = model_label['input_ids']
        return model_data
    
    tok_df = dataset.map(process, batched=True)
    
    tf_train_df = model.prepare_tf_dataset(
        tok_df["train"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=8,
    )
    tf_eval_df = model.prepare_tf_dataset(
        tok_df["validation"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=8,
    )
    
    num_train_epochs = 3
    num_train_steps = len(tf_train_dataset) * num_train_epochs
    model_name = model_checkpoint.split("/")[-1]

    optimizer, schedule = create_optimizer(
        init_lr=5.6e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )

    model.compile(optimizer=optimizer)

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    
    train_results = model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=8)
    
    print("Saving model")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)