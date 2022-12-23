import pandas as pd
import argparse
import os
import tensorflow as tf
from nltk import wordpunct_tokenize
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
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

    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = examples["tag"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tok_df = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    tf_train_df = tok_df["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )
    tf_test_df = tok_df["test"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
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

    model.fit(tf_train_df, epochs=num_epochs)
    model.evaluate(tf_test_df, batch_size=args.test_batch_size, return_dict=True)

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)