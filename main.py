import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset
from scipy.stats import mode

# Set fixed parameters
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
N_SPLITS = 5

# Define ensemble method: choose "soft", "hard", or "stacking"
ENSEMBLE_METHOD = "stacking"  # Options: "soft", "hard", "stacking"

# Define model names; feel free to add more
MODEL_NAMES = {
    "MathBERT": "tbs17/MathBERT",
    "BERT_BASE": "bert-base-uncased"
}

def load_data():
    train_df = pd.read_csv('/home/amma/Documents/stain/math/train.csv')
    test_df = pd.read_csv('/home/amma/Documents/stain/math/test.csv')
    
    def clean_math_text(text):
        # Preserve mathematical notation with a special marker
        text = re.sub(r'\$(.*?)\$', r' [MATH] \1 [MATH] ', text)
        text = re.sub(r'\\\w+', lambda m: ' ' + m.group(0) + ' ', text)
        return text.strip()
    
    train_df['cleaned'] = train_df['Question'].apply(clean_math_text)
    test_df['cleaned'] = test_df['Question'].apply(clean_math_text)
    
    train_df.drop(columns=['Question'], inplace=True)
    test_df.drop(columns=['Question'], inplace=True)
    
    return train_df, test_df

class MathDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels  # labels can be None for inference
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # For MathBERT, add the special token to preserve math notation
    if "MathBERT" in model_name:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[MATH]']})
    return tokenizer

def train_model(model_name, train_texts, train_labels, val_texts, val_labels):
    """Fine-tune a model on the given split and return the trained model and its tokenizer."""
    tokenizer = get_tokenizer(model_name)
    train_dataset = MathDataset(train_texts, train_labels, tokenizer)
    eval_dataset = MathDataset(val_texts, val_labels, tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=8,
        ignore_mismatched_sizes=True
    )
    # If extra tokens are added, resize the embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    args = TrainingArguments(
        output_dir=f'./{model_name.replace("/", "_")}_output',
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        fp16=True,  # if supported
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model='f1_micro',
        report_to="none"
    )
    
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {'f1_micro': f1_score(p.label_ids, preds, average='micro')}
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    return model, tokenizer, trainer

def get_predictions(model, tokenizer, texts, mode="test"):
    """
    Get predictions (logits) from a model given a list of texts.
    If mode="test", we predict on test data; otherwise, on validation data.
    """
    dataset = MathDataset(texts, labels=[0]*len(texts), tokenizer=tokenizer)
    temp_args = TrainingArguments(output_dir='./temp', per_device_eval_batch_size=32, report_to="none")
    temp_trainer = Trainer(model=model, args=temp_args)
    outputs = temp_trainer.predict(dataset)
    return outputs.predictions  # logits

def train_and_ensemble():
    train_df, test_df = load_data()
    test_texts = test_df['cleaned'].tolist()
    test_ids = test_df['id'].values
    n_test = len(test_texts)
    n_train = len(train_df)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # Containers for ensemble predictions for test set per model/fold
    ensemble_test_logits = {model_key: [] for model_key in MODEL_NAMES.keys()}
    ensemble_test_preds = {model_key: [] for model_key in MODEL_NAMES.keys()}  # for hard voting
    
    # For stacking: store OOF meta-features for training and list for test features per fold
    if ENSEMBLE_METHOD == "stacking":
        meta_train = np.zeros((n_train, len(MODEL_NAMES) * 8))  # each model contributes 8 probability values
        meta_test_all = np.zeros((N_SPLITS, n_test, len(MODEL_NAMES) * 8))
    
    fold_no = 0
    for train_index, val_index in skf.split(train_df['cleaned'], train_df['label']):
        fold_no += 1
        print(f"\n=== Fold {fold_no} ===")
        # Split train/validation for this fold
        tr_texts = train_df.iloc[train_index]['cleaned'].tolist()
        tr_labels = train_df.iloc[train_index]['label'].tolist()
        val_texts = train_df.iloc[val_index]['cleaned'].tolist()
        val_labels = train_df.iloc[val_index]['label'].tolist()
        
        # For each model, fine-tune on the current fold and collect predictions
        for model_key, model_name in MODEL_NAMES.items():
            print(f"Training model: {model_key}")
            model, tokenizer, trainer = train_model(model_name, tr_texts, tr_labels, val_texts, val_labels)
            
            # Get predictions on validation set (OOF predictions) as logits
            val_logits = get_predictions(model, tokenizer, val_texts, mode="val")
            # Also get test predictions (logits)
            test_logits = get_predictions(model, tokenizer, test_texts, mode="test")
            
            # For soft voting we accumulate logits; for hard voting, we accumulate argmax predictions
            ensemble_test_logits[model_key].append(test_logits)
            ensemble_test_preds[model_key].append(np.argmax(test_logits, axis=1))
            
            if ENSEMBLE_METHOD == "stacking":
                # For stacking, use softmax probabilities as meta-features.
                # Here we use a simple normalization: apply softmax on logits.
                exp_val = np.exp(val_logits)
                val_probs = exp_val / np.expand_dims(np.sum(exp_val, axis=1), axis=1)
                exp_test = np.exp(test_logits)
                test_probs = exp_test / np.expand_dims(np.sum(exp_test, axis=1), axis=1)
                
                # Determine feature indices for this model in the meta-feature vector.
                col_start = list(MODEL_NAMES.keys()).index(model_key) * 8
                col_end = col_start + 8
                # Save OOF predictions for validation samples.
                meta_train[val_index, col_start:col_end] = val_probs
                # Save test predictions for this fold.
                meta_test_all[fold_no-1, :, col_start:col_end] = test_probs
            
            # Cleanup to free GPU memory
            del model
            torch.cuda.empty_cache()
    
    # Now ensemble the test set predictions based on the selected strategy
    if ENSEMBLE_METHOD == "soft":
        # Soft Voting: Average logits across folds and models, then take argmax.
        aggregated_logits = None
        for model_key, logits_list in ensemble_test_logits.items():
            model_avg_logits = np.mean(np.array(logits_list), axis=0)
            if aggregated_logits is None:
                aggregated_logits = model_avg_logits
            else:
                aggregated_logits += model_avg_logits
        aggregated_logits /= len(MODEL_NAMES)
        final_preds = np.argmax(aggregated_logits, axis=1)
    
    elif ENSEMBLE_METHOD == "hard":
        # Hard Voting: For each model, average its fold predictions (majority vote across folds) first,
        # then take majority vote across models.
        all_preds = []
        for model_key, preds_list in ensemble_test_preds.items():
            # For each model, concatenate the predictions from each fold and then compute mode along the fold axis.
            model_preds = np.array(preds_list)  # shape: (n_splits, n_test)
            # Majority vote across folds for this model:
            model_vote, _ = mode(model_preds, axis=0)
            all_preds.append(model_vote.flatten())
        # Now, majority vote across models:
        all_preds_array = np.array(all_preds)  # shape: (n_models, n_test)
        final_vote, _ = mode(all_preds_array, axis=0)
        final_preds = final_vote.flatten()
    
    elif ENSEMBLE_METHOD == "stacking":
        # Stacking: Build meta-model using out-of-fold predictions.
        # Average the test meta-features over folds
        meta_test = np.mean(meta_test_all, axis=0)  # shape: (n_test, n_models*8)
        
        # Train a logistic regression meta-model on meta_train features
        meta_model = LogisticRegression(max_iter=1000, random_state=SEED)
        # meta_train now has OOF predictions; we train on these using the true train labels.
        meta_model.fit(meta_train, train_df['label'].values)
        final_preds = meta_model.predict(meta_test)
    
    else:
        raise ValueError("Invalid ENSEMBLE_METHOD selected. Choose from 'soft', 'hard', or 'stacking'.")
    
    submission = pd.DataFrame({
        'id': test_ids,
        'label': final_preds
    })
    submission.to_csv('submission_stacking.csv', index=False)
    print("Submission file saved as submission.csv")
    return submission

if __name__ == "__main__":
    submission = train_and_ensemble()
    print(submission.head())
