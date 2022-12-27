import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from transformers.keras_callbacks import KerasMetricCallback
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, create_optimizer, AdamWeightDecay

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
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    tf_train_df = tok_df['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask', 'labels'],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=8,
    )
    tf_test_df = tok_df['test'].to_tf_dataset(
        columns=['input_ids', 'attention_mask', 'labels'],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=8,
    )
    tf_eval_df = tok_df['validation'].to_tf_dataset(
        columns=['input_ids', 'attention_mask', 'labels'],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=8,
    )

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    model.compile(optimizer=optimizer)

    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_df)
    
    train_results = model.fit(tf_train_df, validation_data=tf_eval_df, epochs=args.epochs, callbacks=[metric_callback])
    eval_results = model.evaluate(tf_test_df, return_dict=True)
    
    print("Saving model")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)