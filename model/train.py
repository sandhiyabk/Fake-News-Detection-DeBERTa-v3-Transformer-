import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# Relative import of preprocess from the same directory
from model.preprocess import preprocess_dataframe

MODEL_NAME = "microsoft/deberta-v3-small"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 3

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1-score for evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from torch.nn import CrossEntropyLoss

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Move class weights to the same device as labels
            weights = self.class_weights.to(labels.device)
            loss_fct = CrossEntropyLoss(weight=weights)
        else:
            loss_fct = CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_model(csv_path=None, test_size=0.2):
    """
    Load data, preprocess, and fine-tune DeBERTa-v3 model.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback to example dataset from HF
        print("No local CSV provided. Loading 'gonzalo-nm/fake-news' from HF datasets.")
        from datasets import load_dataset
        dataset = load_dataset("gonzalo-nm/fake-news")
        df = pd.DataFrame(dataset['train'])
    
    # Map labels: 1 = Fake, 0 = Real? 
    # Usually: 0: Fake, 1: Real
    # Ensure standard labels: 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    # 2. Preprocess Data
    from model.preprocess import preprocess_dataframe, get_class_weights
    df = preprocess_dataframe(df)
    
    # Calculate class weights for imbalance
    weights_dict = get_class_weights(df)
    # Convert dict to tensor ordered by label ID [0, 1]
    class_weights = torch.tensor([weights_dict[0], weights_dict[1]], dtype=torch.float)
    
    # 3. Train/Validation Split (80/20)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    
    # Convert to HF Dataset object
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    # 4. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # 5. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )
    
    # 7. Trainer with Early Stopping and Custom Weighted Loss
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights
    )
    
    # 8. Train
    print("Starting training...")
    trainer.train()
    
    # 9. Evaluate
    print("Evaluating...")
    print(trainer.evaluate())
    
    # 10. Save Model and Tokenizer
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return trainer

if __name__ == "__main__":
    # Example usage
    # train_model("data/news.csv")
    train_model()
