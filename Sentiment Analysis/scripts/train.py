import pandas as pd
from datasets import Dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
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
    df = pd.read_csv('s3://sagemaker-us-east-1-179099335435/dataset/review.csv')
    df = df.dropna(axis=0, subset=['reviewText'])
    df['sentiment'] = df['overall'].map({1:0, 2:0, 3:1, 4:1, 5:1})
    df = df[["sentiment", "reviewText"]]
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    
    def tokenization(review):
        return tokenizer(review["reviewText"], truncation=True)
    
    tok_df = dataset.map(tokenization, batched=True)
    
    tf_train_df = tok_df["train"].to_tf_dataset(
                            columns=['attention_mask', 'input_ids', 'token_type_ids'],
                            label_cols=['sentiment'],
                            shuffle=False,
                            collate_fn=data_collator,
                            batch_size=8)
    tf_test_df = tok_df["test"].to_tf_dataset(
                            columns=['attention_mask', 'input_ids', 'token_type_ids'],
                            label_cols=['sentiment'],
                            shuffle=False,
                            collate_fn=data_collator,
                            batch_size=8)
    
    num_train_steps = len(tf_train_df) * args.epochs
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
    )
    
    opt = Adam(learning_rate=lr_scheduler)
    
    model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    train_results = model.fit(tf_train_df, epochs=args.epochs, batch_size=args.train_batch_size)
    
    eval_results = model.evaluate(tf_test_df, batch_size=args.test_batch_size, return_dict=True)
    
    print("Saving model")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)