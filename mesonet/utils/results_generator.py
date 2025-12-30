import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import json

def generate_visualizations(true_labels, predicted_labels, confidences, raw_scores, output_dir, model_name="MesoNet"):
    """
    Generate 5 academic-quality visualizations and save results.
    
    Args:
        true_labels: List of ground truth labels ('real' or 'fake')
        predicted_labels: List of predicted labels ('real' or 'fake')
        confidences: List of confidence scores (0.5 to 1.0)
        raw_scores: List of raw probabilities of being fake (0.0 to 1.0)
        output_dir: Directory to save results
        model_name: Name of model for titles
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Convert labels to binary (fake=1, real=0)
    y_true = [1 if l == 'fake' else 0 for l in true_labels]
    y_pred = [1 if l == 'fake' else 0 for l in predicted_labels]
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix\n{model_name}', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. Metrics Bar Chart
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#9b59b6', '#f1c40f'])
    plt.ylim(0, 1.1)
    plt.title(f'Performance Metrics\n{model_name}', pad=20)
    plt.ylabel('Score')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_bar_chart.png'))
    plt.close()
    
    # 3. Prediction Distribution
    plt.figure(figsize=(8, 6))
    # Count predictions
    pred_counts = {'Real': predicted_labels.count('real'), 'Fake': predicted_labels.count('fake')}
    plt.bar(pred_counts.keys(), pred_counts.values(), color=['#3498db', '#e74c3c'])
    plt.title(f'Prediction Distribution\n{model_name}', pad=20)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()
    
    # 4. Confidence Distribution
    plt.figure(figsize=(10, 6))
    
    # Separate confidences by prediction
    real_conf = [c for p, c in zip(predicted_labels, confidences) if p == 'real']
    fake_conf = [c for p, c in zip(predicted_labels, confidences) if p == 'fake']
    
    if real_conf:
        sns.histplot(real_conf, color='#3498db', label='Predicted Real', kde=True, bins=10, alpha=0.6)
    if fake_conf:
        sns.histplot(fake_conf, color='#e74c3c', label='Predicted Fake', kde=True, bins=10, alpha=0.6)
        
    plt.title(f'Confidence Score Distribution\n{model_name}', pad=20)
    plt.xlabel('Confidence Score (0.5 - 1.0)')
    plt.ylabel('Count')
    plt.xlim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # 5. ROC Curve
    if len(set(y_true)) > 1: # Only plot ROC if we have both classes
        fpr, tpr, _ = roc_curve(y_true, raw_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic\n{model_name}', pad=20)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
    
    # Save results to JSON
    results_data = {
        'model': model_name,
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'counts': {
            'total': len(y_true),
            'real_ground_truth': y_true.count(0),
            'fake_ground_truth': y_true.count(1),
            'real_predicted': y_pred.count(0),
            'fake_predicted': y_pred.count(1)
        }
    }
    
    return metrics
