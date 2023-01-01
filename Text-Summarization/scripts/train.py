import os
os.system("pip install -U sagemaker")
os.system("pip install -U transformers")
import boto3
import argparse
import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Sequence, Value
from sagemaker.session import Session
from sagemaker.experiments import load_run
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
    boto_session = boto3.session.Session(region_name='us-east-1')
    session = Session(boto_session=boto_session)
    
    rouge = evaluate.load("rouge")

    print("Read Training Data")
    dataset = load_dataset('kmfoda/booksum')
    
    model_checkpoint = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
    generate_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

    prefix = "summarize: "
    def process(example):
        inputs = [prefix + doc for doc in example["chapter"]]
        model_data = tokenizer(example['chapter'], max_length=1024, truncation=True)
        with tokenizer.as_target_tokenizer():
            model_label = tokenizer(example["summary"], max_length=128, truncation=True)
        model_data['labels'] = model_label['input_ids'] 
        return model_data
    
    tok_df = dataset.map(process, batched=True)
    tf_train_df = model.prepare_tf_dataset(
        tok_df["train"],
        collate_fn=collator,
        shuffle=True,
        batch_size=8,
    )
    tf_test_df = model.prepare_tf_dataset(
        tok_df["test"],
        collate_fn=collator,
        shuffle=False,
        batch_size=8,
    )
    tf_eval_df = model.prepare_tf_dataset(
        tok_df["validation"],
        collate_fn=collator,
        shuffle=False,
        batch_size=8,
    )
    generate_df = model.prepare_tf_dataset(
        tok_df["validation"],
        collate_fn=generate_collator,
        shuffle=False,
        batch_size=8,
    )
    
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer)
    
    def metric_fn(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
                          
    metric_callback = KerasMetricCallback( metric_fn, eval_dataset=generate_df, predict_with_generate=True, use_xla_generation=True)
    
    with load_run(sagemaker_session=session) as run:
        train_results = model.fit(tf_train_df, validation_data=tf_eval_df, epochs=1, callbacks=[metric_callback])
        
        print(f'Train \n {train_results}')
        print(f'Test \n {eval_results}')

        print("Saving model")
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)