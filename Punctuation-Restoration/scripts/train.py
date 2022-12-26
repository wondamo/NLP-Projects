import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
from nltk import wordpunct_tokenize
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification, create_optimizer
from datasets import ClassLabel, Sequence, Dataset

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

    lab2id = {',':1,'.':2,'!':3,'NaN':0}
    id2lab = {v:k for k,v in lab2id.items()}

    df = pd.read_csv('review-Copy1.csv')
    df = df.dropna(axis=0, subset=['reviewText'])
    df = df[["reviewText"]]
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.train_test_split(test_size=0.2)

    def process(text):
        lab, tokens = [], []
        tok = wordpunct_tokenize(text['reviewText'])
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
    dataset.features['tag']=Sequence(feature=ClassLabel(num_classes=4, names=['Nan', ',', '.', '!']))
    dataset = dataset.train_test_split(test_size=0.2)

    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2lab, label2id=lab2id)
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

    tok_df = dataset.map(tokenize_and_align_labels, batched=True)
    
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
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    tf_train_df = model.prepare_tf_dataset(
        tok_df['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=16,
    )
    tf_test_df = model.prepare_tf_dataset(
        tok_df['test'],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=16,
    )

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    num_epochs = 3
    num_train_steps = len(tf_train_df) * num_epochs

    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)
    
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_df)

    model.fit(tf_train_df, epochs=num_epochs)

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)