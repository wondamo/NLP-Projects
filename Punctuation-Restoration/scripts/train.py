import os
import argparse
import numpy as np
import pandas as pd
import evaluate
import tensorflow as tf
from nltk import wordpunct_tokenize
from sagemaker.session import Session
from sagemaker.experiments import load_run
from transformers.keras_callbacks import KerasMetricCallback
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification, create_optimizer
from datasets import ClassLabel, Sequence, Dataset

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args, _ = parser.parse_known_args()
#     sagemaker_session = Session()
    
    seqeval = evaluate.load("seqeval")

    lab2id = {',':1,'.':2,'!':3, '?':4, "'":5,'NaN':0}
    id2lab = {v:k for k,v in lab2id.items()}

    dataset = load_dataset('yelp_polarity')
    df = dataset['train'].train_test_split(test_size=0.1)
    dataset['validation'] = df['test']
    dataset['train'] = df['train']

    def process(text):
        lab, tokens = [], []
        tok = wordpunct_tokenize(text['text'])
        if len(tok) != 1:
            while tok[0] in lab2id.keys():
                del tok[0]
            for i in range(len(tok)):
                if tok[i] in lab2id.keys():
                    lab[-1] = lab2id[tok[i]]
                else:
                    lab.append(0)
                    tokens.append(tok[i])
        return {'tokens':tokens, 'tag':lab}

    dataset = dataset.map(process)
    dataset = dataset.filter(lambda example: len(example['tokens']) != 0)
    dataset['train'].features['tag']=Sequence(feature=ClassLabel(num_classes=6, names=['Nan', ',', '.', '!', '?', "'"]))
    dataset['test'].features['tag']=Sequence(feature=ClassLabel(num_classes=6, names=['Nan', ',', '.', '!', '?', "'"]))
    
    label_list=dataset['train'].features['tag'].feature.names

    model_checkpoint = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=4, id2label=id2lab, label2id=lab2id)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
        labels = []
        for i, label in enumerate(examples["tag"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                previous_word = word_id
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tok_df = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset['train'].column_names)
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        print(results)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    tf_train_df = tok_df['train'].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )
    tf_test_df = tok_df['test'].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )
    tf_val_df = tok_df['validation'].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    num_train_steps = len(tf_train_df) * args.epochs

    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)
    
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_val_df)
    
    with load_run() as run:
        train_results = model.fit(tf_train_df, validation_data=tf_val_df, epochs=args.epochs, callbacks=[metric_callback])
        eval_results = model.evaluate(tf_test_df, return_dict=True)
        
        print(f'Train \n {train_results}')
        print(f'Test \n {eval_results}')
        
        run.log_metric(name="Train Loss", value=train_results[0])
        run.log_metric(name="Train Accuracy", value=train_results[1])
        run.log_metric(name="Test Loss", value=test_results[0])
        run.log_metric(name="Test Accuracy", value=test_results[1])

        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)