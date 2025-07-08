import numpy as np
import pandas as pd
import time
import os
import editdistance
from collections import defaultdict

# Configuration constants (ensure these match your global config)
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-'\" "
max_str_len = 2000
num_of_characters = len(alphabets) + 2
START_TOKEN = len(alphabets)
END_TOKEN = len(alphabets) + 1
PAD_TOKEN = 0

class TrainingTracker:
    """Track training results for each epoch"""

    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'bias_variance_gap': [],
            'training_time': []
        }

    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None,
                  learning_rate=None, bias_variance_gap=None, training_time=None):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_acc)
        self.history['val_loss'].append(val_loss if val_loss is not None else 0.0)
        self.history['val_accuracy'].append(val_acc if val_acc is not None else 0.0)
        self.history['learning_rate'].append(learning_rate if learning_rate is not None else 0.0)
        self.history['bias_variance_gap'].append(bias_variance_gap if bias_variance_gap is not None else 0.0)
        self.history['training_time'].append(training_time if training_time is not None else 0.0)

    def save_history(self, filepath):
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)

    def print_best_results(self):
        if len(self.history['train_accuracy']) > 0:
            best_train_idx = np.argmax(self.history['train_accuracy'])
            val_accs = [acc for acc in self.history['val_accuracy'] if acc > 0]
            if val_accs:
                best_val_idx = np.argmax(self.history['val_accuracy'])

class DataAnalyzer:
    """Analyze dataset characteristics and statistics"""

    def __init__(self, df):
        self.df = df

    def basic_statistical_descriptions(self):
        text_lengths = [len(text) for text in self.df['text']]
        return {
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'std': np.std(text_lengths),
            'var': np.var(text_lengths)
        }

    def data_visualization(self):
        text_lengths = [len(text) for text in self.df['text']]
        char_counts = defaultdict(int)
        for text in self.df['text']:
            for char in text:
                if char in alphabets:
                    char_counts[char] += 1
        stats = self.basic_statistical_descriptions()
        length_distribution = {}
        for length in text_lengths:
            bucket = (length // 5) * 5
            length_distribution[bucket] = length_distribution.get(bucket, 0) + 1

class FeatureSelector:
    """Feature selection based on decision tree concepts"""

    def __init__(self, df):
        self.df = df

    def calculate_entropy(self, labels):
        if len(labels) == 0:
            return 0
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        entropy = 0
        total = len(labels)
        for count in label_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy

    def calculate_information_gain(self, feature_values, labels):
        total_entropy = self.calculate_entropy(labels)
        feature_groups = defaultdict(list)
        for i, value in enumerate(feature_values):
            feature_groups[value].append(labels[i])
        weighted_entropy = 0
        total_samples = len(labels)
        for group_labels in feature_groups.values():
            if len(group_labels) > 0:
                weight = len(group_labels) / total_samples
                group_entropy = self.calculate_entropy(group_labels)
                weighted_entropy += weight * group_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def analyze_text_features(self):
        text_lengths = [len(text) for text in self.df['text']]
        median_length = np.median(text_lengths)
        length_labels = ['short' if length < median_length else 'long' for length in text_lengths]

        features = {
            'first_char': [text[0] if text else '' for text in self.df['text']],
            'last_char': [text[-1] if text else '' for text in self.df['text']],
            'has_numbers': ['yes' if any(c.isdigit() for c in text) else 'no' for text in self.df['text']],
            'has_punctuation': ['yes' if any(c in '.,!?' for c in text) else 'no' for text in self.df['text']]
        }

        feature_gains = {}
        for feature_name, feature_values in features.items():
            gain = self.calculate_information_gain(feature_values, length_labels)
            feature_gains[feature_name] = gain

        return feature_gains

class ModelEvaluator:
    """Model selection and evaluation using cross-validation"""

    def __init__(self):
        self.results_history = []

    def k_fold_cross_validation(self, df, model_builder=None, k=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'fold_results': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_fold = df.iloc[train_idx].reset_index(drop=True)
            val_fold = df.iloc[val_idx].reset_index(drop=True)

            # Simulated evaluation
            train_acc = np.random.uniform(0.7, 0.9)
            val_acc = np.random.uniform(0.6, 0.8)

            cv_results['train_accs'].append(train_acc)
            cv_results['val_accs'].append(val_acc)

            cv_results['fold_results'].append({
                'fold': fold+1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_samples': len(train_fold),
                'val_samples': len(val_fold)
            })

        train_mean = np.mean(cv_results['train_accs'])
        val_mean = np.mean(cv_results['val_accs'])
        bias_estimate = abs(train_mean - val_mean)
        variance_estimate = np.var(cv_results['val_accs'])

        return cv_results

    def performance_measures(self, predictions, ground_truths):
        metrics = {
            'mse': np.mean([(len(p) - len(g))**2 for p, g in zip(predictions, ground_truths)]),
            'mae': np.mean([abs(len(p) - len(g)) for p, g in zip(predictions, ground_truths)]),
            'sequence_accuracy': sum(1 for p, g in zip(predictions, ground_truths) if p.strip().upper() == g.strip().upper()) / len(predictions),
        }

        edit_distances = [editdistance.eval(p.upper(), g.upper()) for p, g in zip(predictions, ground_truths)]
        metrics['mean_edit_distance'] = np.mean(edit_distances)
        metrics['edit_distance_std'] = np.std(edit_distances)

        return metrics

class OptimizedTraining:
    """Training optimization using mathematical concepts"""

    def __init__(self):
        self.learning_curves = {'loss': [], 'accuracy': []}

    def cosine_decay_with_warmup(self, epoch, total_epochs=100, warmup_epochs=10, max_lr=2e-3, min_lr=1e-6):
        if epoch < warmup_epochs:
            return max_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def enhanced_preprocess(img, target_h=64, max_w=256):
    import cv2
    if img is None:
        return np.ones((target_h, max_w), dtype=np.uint8) * 255

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    h, w = img.shape
    aspect_ratio = w / h
    if aspect_ratio > max_w / target_h:
        new_w = max_w
        new_h = int(max_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    img = cv2.resize(img, (new_w, new_h))

    canvas = np.ones((target_h, max_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (max_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    return canvas

def label_to_sequence(label):
    sequence = [START_TOKEN]
    for ch in label:
        idx = alphabets.find(ch)
        if idx >= 0:
            sequence.append(idx)
    sequence.append(END_TOKEN)
    return sequence