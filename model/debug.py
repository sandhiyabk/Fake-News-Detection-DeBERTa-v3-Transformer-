import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model.predict import predict_news
from model.preprocess import preprocess_dataframe

def evaluate_on_test_set(csv_path):
    """
    Evaluates the model on a provided test dataset and visualizes performance.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df)
    
    # 1. Predictions
    print(f"Running predictions on {len(df)} samples...")
    results = []
    for text in df['text']:
        pred = predict_news(text)
        results.append(1 if pred['label'] == "Real" else 0)
        
    y_true = df['label'].values
    y_pred = np.array(results)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Fake News Detection')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to confusion_matrix.png")
    
    # 3. Sample Errors
    print("\n--- ERROR ANALYSIS (Sample mismatches) ---")
    mismatches = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            mismatches.append((df['text'].iloc[i], true, pred))
            
    # Print first 5 errors
    for i, (text, true, pred) in enumerate(mismatches[:5]):
        true_label = "Real" if true == 1 else "Fake"
        pred_label = "Real" if pred == 1 else "Fake"
        print(f"\nSample Error #{i+1}:")
        print(f"Text snippet: {text[:200]}...")
        print(f"Actual: {true_label} | Predicted: {pred_label}")
        
    # 4. Detailed Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

def adjust_threshold(y_true, y_scores, threshold=0.5):
    """
    Tune the classification threshold for better performance.
    """
    y_pred = (y_scores >= threshold).astype(int)
    print(f"\nResults with threshold {threshold}:")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
    return y_pred

if __name__ == "__main__":
    # evaluate_on_test_set("data/test_news.csv")
    pass
