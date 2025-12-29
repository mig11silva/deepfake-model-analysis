"""
Results Generator - Creates metrics and visualizations from predictions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)


# Configuration
BASE_DIR = Path(__file__).parent.parent.absolute()
RESULTS_DIR = BASE_DIR / "results"
FIGURE_DPI = 150


class ResultsGenerator:
    """Generates metrics and visualizations from model predictions."""
    
    def __init__(self):
        """Initialize the generator."""
        self.output_dir = RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def _prepare_data(self, predictions: List[Dict]) -> Tuple[List[str], List[str], List[float]]:
        """Extract ground truth, predictions, and confidence scores."""
        ground_truths, pred_labels, confidences = [], [], []
        
        for pred in predictions:
            if pred.get('prediction') != 'error':
                ground_truths.append(pred['ground_truth'])
                pred_labels.append(pred['prediction'])
                confidences.append(pred['confidence'])
        
        return ground_truths, pred_labels, confidences
    
    def _calculate_metrics(self, ground_truths: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            'accuracy': round(accuracy_score(ground_truths, predictions), 4),
            'precision': round(precision_score(ground_truths, predictions, pos_label='fake', zero_division=0), 4),
            'recall': round(recall_score(ground_truths, predictions, pos_label='fake', zero_division=0), 4),
            'f1_score': round(f1_score(ground_truths, predictions, pos_label='fake', zero_division=0), 4)
        }
    
    def _save_confusion_matrix(self, ground_truths: List[str], predictions: List[str]) -> None:
        """Generate confusion matrix."""
        labels = ['fake', 'real']
        cm = confusion_matrix(ground_truths, predictions, labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={'size': 20})
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=FIGURE_DPI)
        plt.close()
        print("  ✓ confusion_matrix.png")
    
    def _save_metrics_chart(self, metrics: Dict[str, float]) -> None:
        """Generate metrics bar chart."""
        names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, values, color=colors, edgecolor='black')
        
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Classification Metrics', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_bar_chart.png", dpi=FIGURE_DPI)
        plt.close()
        print("  ✓ metrics_bar_chart.png")
    
    def _save_prediction_distribution(self, predictions: List[str]) -> None:
        """Generate prediction distribution chart."""
        fake_count = predictions.count('fake')
        real_count = predictions.count('real')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(['Fake', 'Real'], [fake_count, real_count], 
                      color=['#E53935', '#43A047'], edgecolor='black')
        
        for bar, count in zip(bars, [fake_count, real_count]):
            ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Prediction Distribution', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_distribution.png", dpi=FIGURE_DPI)
        plt.close()
        print("  ✓ prediction_distribution.png")
    
    def _save_confidence_distribution(self, confidences: List[float], predictions: List[str]) -> None:
        """Generate confidence histogram."""
        fake_conf = [c for c, p in zip(confidences, predictions) if p == 'fake']
        real_conf = [c for c, p in zip(confidences, predictions) if p == 'real']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(fake_conf, bins=20, alpha=0.7, label='Fake', color='#E53935', edgecolor='black')
        ax.hist(real_conf, bins=20, alpha=0.7, label='Real', color='#43A047', edgecolor='black')
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Confidence Distribution', fontsize=16, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confidence_distribution.png", dpi=FIGURE_DPI)
        plt.close()
        print("  ✓ confidence_distribution.png")
    
    def _save_roc_curve(self, ground_truths: List[str], predictions: List[str], confidences: List[float]) -> None:
        """Generate ROC curve."""
        y_true = [1 if g == 'fake' else 0 for g in ground_truths]
        y_scores = [c if p == 'fake' else 1-c for p, c in zip(predictions, confidences)]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=FIGURE_DPI)
        plt.close()
        print("  ✓ roc_curve.png")
    
    def generate_all(self, predictions: List[Dict]) -> Dict[str, float]:
        """Generate all visualizations and metrics."""
        print(f"\nGenerating results...")
        print(f"Output: {self.output_dir}")
        print("-" * 40)
        
        ground_truths, pred_labels, confidences = self._prepare_data(predictions)
        metrics = self._calculate_metrics(ground_truths, pred_labels)
        
        print(f"\n  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1-Score:  {metrics['f1_score']:.2%}\n")
        
        self._save_confusion_matrix(ground_truths, pred_labels)
        self._save_metrics_chart(metrics)
        self._save_prediction_distribution(pred_labels)
        self._save_confidence_distribution(confidences, pred_labels)
        self._save_roc_curve(ground_truths, pred_labels, confidences)
        
        print("-" * 40)
        return metrics
