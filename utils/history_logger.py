import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import numpy as np
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryLogger:
    """
    OOP compliant HistoryLogger to automatically log training performance history.
    Automatically records evaluation metrics and flattened confusion matrices to a CSV file.
    """
    def __init__(self, log_dir: str) -> None:
        """
        Initialize HistoryLogger.

        Args:
            log_dir (str): Path to directory where logs and CSV files are stored.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_csv_path = os.path.join(self.log_dir, 'history.csv')

        # Create history.csv and write headers if the file does not exist
        if not os.path.exists(self.history_csv_path):
            with open(self.history_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Column headers including flattened confusion matrix elements
                headers = [
                    'epoch', 
                    'val_loss', 
                    'val_accuracy', 
                    'val_mAP',
                    'val_auc_class_none', 'val_auc_class_strong', 'val_auc_class_medium', 'val_auc_class_weak',
                    'val_ap_class_none', 'val_ap_class_strong', 'val_ap_class_medium', 'val_ap_class_weak',
                    
                    # 16 flattened confusion matrix columns (Actual vs Predicted)
                    'cm_none_none', 'cm_none_strong', 'cm_none_medium', 'cm_none_weak',
                    'cm_strong_none', 'cm_strong_strong', 'cm_strong_medium', 'cm_strong_weak',
                    'cm_medium_none', 'cm_medium_strong', 'cm_medium_medium', 'cm_medium_weak',
                    'cm_weak_none', 'cm_weak_strong', 'cm_weak_medium', 'cm_weak_weak'
                ]
                writer.writerow(headers)

    def _save_confusion_matrix_csv(self, path: str, matrix: np.ndarray) -> None:
        """
        Private Method to save 2D confusion matrix to a labeled CSV file.
        """
        labels = ['none', 'strong', 'medium', 'weak']
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header row (predicted labels)
            writer.writerow(['Actual\\Predicted'] + labels)
            # Write rows (actual labels)
            for idx, label in enumerate(labels):
                writer.writerow([label] + list(matrix[idx]))

    def log_epoch(self, epoch: int, val_loss: float, val_statistics: dict, is_best: bool = False) -> None:
        """
        Record performance metrics for the current epoch and append to the history CSV file.

        Args:
            epoch (int): Current epoch number.
            val_loss (float): Validation loss value.
            val_statistics (dict): Dictionary returned by AudioEvaluator.
            is_best (bool): True if current epoch represents the best validation performance. Default: False.
        """
        val_acc = np.mean(val_statistics['accuracy'])
        val_mAP = np.mean(val_statistics['average_precision'])
        val_auc = val_statistics['auc']               # Class AUC array
        val_ap = val_statistics['average_precision']   # Class AP array
        cm = val_statistics['confu_matrix']           # 2D confusion matrix

        # Flatten 2D confusion matrix to 1D list
        cm_flat = list(cm.flatten())

        # Write new row to history.csv
        row_data = [
            epoch,
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_mAP:.6f}",
            f"{val_auc[0]:.6f}", f"{val_auc[1]:.6f}", f"{val_auc[2]:.6f}", f"{val_auc[3]:.6f}",
            f"{val_ap[0]:.6f}", f"{val_ap[1]:.6f}", f"{val_ap[2]:.6f}", f"{val_ap[3]:.6f}"
        ] + [int(val) for val in cm_flat]

        with open(self.history_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Update best confusion matrix CSV if this is the best epoch
        if is_best:
            best_cm_path = os.path.join(self.log_dir, 'confusion_matrix_best.csv')
            self._save_confusion_matrix_csv(best_cm_path, cm)

    def save_summary(self, training_time: float, inference_time_ms: float, val_statistics: dict, test_statistics: dict) -> None:
        """
        Save final summary report (Val & Test performance metrics, training and inference times) to summary.csv.
        """
        summary_csv_path = os.path.join(self.log_dir, 'summary.csv')
        
        # Check if file exists to write headers
        file_exists = os.path.exists(summary_csv_path)
        
        headers = [
            'Training Time (s)',
            'Inference Time (ms/sample)',
            'Precision Val (Weighted)', 'Recall Val (Weighted)', 'F1-score Val (Weighted)', 'Accuracy Val',
            'Precision Val (Macro)', 'Recall Val (Macro)', 'F1-score Val (Macro)',
            'Precision Test (Weighted)', 'Recall Test (Weighted)', 'F1-score Test (Weighted)', 'Accuracy Test',
            'Precision Test (Macro)', 'Recall Test (Macro)', 'F1-score Test (Macro)'
        ]
        
        row_data = [
            f"{training_time:.2f}",
            f"{inference_time_ms:.3f}",
            f"{val_statistics['prec_weighted']:.6f}",
            f"{val_statistics['rec_weighted']:.6f}",
            f"{val_statistics['f1_weighted']:.6f}",
            f"{val_statistics['accuracy']:.6f}",
            f"{val_statistics['prec_macro']:.6f}",
            f"{val_statistics['rec_macro']:.6f}",
            f"{val_statistics['f1_macro']:.6f}",
            f"{test_statistics['prec_weighted']:.6f}",
            f"{test_statistics['rec_weighted']:.6f}",
            f"{test_statistics['f1_weighted']:.6f}",
            f"{test_statistics['accuracy']:.6f}",
            f"{test_statistics['prec_macro']:.6f}",
            f"{test_statistics['rec_macro']:.6f}",
            f"{test_statistics['f1_macro']:.6f}"
        ]
        
        with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row_data)
        logger.info(f"Successfully exported Summary Report to: '{summary_csv_path}'")
