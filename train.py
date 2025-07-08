import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import build_academic_mobilenet_model
from data_generator import AcademicDataGenerator
from utils import TrainingTracker, DataAnalyzer, FeatureSelector, ModelEvaluator, load_academic_optimized_data, create_academic_dummy_data
from optimization import OptimizedTraining  # If you prefer to separate training utilities

def train_academic_complete_model():
    """Train complete OCR model with academic methodologies"""

    # Initialize training tracker
    tracker = TrainingTracker()

    # 1. Load and analyze data
    df = load_academic_optimized_data()
    if df is None or len(df) < 2:
        df = create_academic_dummy_data()
        if len(df) == 0:
            print("No training data available.")
            return None

    # 2. Data analysis (optional, can be commented out if heavy)
    data_analyzer = DataAnalyzer(df)
    stats = data_analyzer.basic_statistical_descriptions()
    data_analyzer.data_visualization()

    # 3. Feature selection (optional)
    feature_selector = FeatureSelector(df)
    feature_gains = feature_selector.analyze_text_features()

    # 4. Model evaluation setup
    evaluator = ModelEvaluator()

    # Create data generators
    train_generator = AcademicDataGenerator(
        df, batch_size=32, train_mode=True, augment=True
    )

    val_generator = AcademicDataGenerator(
        df, batch_size=32, train_mode=False, augment=False
    )

    if len(train_generator) == 0:
        print("No valid training samples.")
        return None

    # 5. Build model
    model = build_academic_mobilenet_model(input_shape=(64, 256, 1))

    # 6. Setup optimizer and callbacks
    optimizer_trainer = OptimizedTraining()
    callbacks = optimizer_trainer.advanced_gradient_descent(
        model, train_generator, val_generator, epochs=80
    )

    best_accuracy = 0
    total_epochs = 80

    # Training loop
    for epoch in range(total_epochs):
        epoch_start_time = time.time()

        # Update learning rate dynamically
        current_lr = optimizer_trainer.cosine_decay_with_warmup(epoch, total_epochs=total_epochs)
        model.optimizer.learning_rate.assign(current_lr)

        epoch_losses = []
        epoch_accs = []
        max_batches = len(train_generator)

        # Training phase
        for batch_idx in range(max_batches):
            try:
                inputs, targets = train_generator[batch_idx]
                history = model.train_on_batch(inputs, targets)
                epoch_losses.append(history[0])
                epoch_accs.append(history[1])
            except Exception as e:
                print(f"Batch {batch_idx} training error: {e}")
                continue

        if epoch_losses:
            avg_train_loss = np.mean(epoch_losses)
            avg_train_acc = np.mean(epoch_accs)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            bias_variance_gap = 0.0

            if len(val_generator) > 0:
                try:
                    val_losses = []
                    val_accs = []
                    eval_batches = min(5, len(val_generator))

                    for val_batch_idx in range(eval_batches):
                        inputs, targets = val_generator[val_batch_idx]
                        val_history = model.test_on_batch(inputs, targets)
                        val_losses.append(val_history[0])
                        val_accs.append(val_history[1])

                    val_loss = np.mean(val_losses)
                    val_acc = np.mean(val_accs)
                    bias_variance_gap = abs(avg_train_acc - val_acc)

                except Exception as e:
                    print(f"Validation error: {e}")

            epoch_time = time.time() - epoch_start_time

            # Log epoch results
            tracker.log_epoch(
                epoch=epoch+1,
                train_loss=avg_train_loss,
                train_acc=avg_train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=current_lr,
                bias_variance_gap=bias_variance_gap,
                training_time=epoch_time
            )

            print(f"Epoch {epoch+1}/{total_epochs} — "
                  f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")

            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                save_path = '/content/drive/MyDrive/academic_complete_model_best_small.keras'
                model.save(save_path)
                print(f"New best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")

        train_generator.on_epoch_end()

    # Save training history and print summary
    history_save_path = '/content/drive/MyDrive/final_training_history.csv'
    tracker.save_history(history_save_path)
    tracker.print_best_results()

    # Optional: perform cross-validation
    cv_results = evaluator.k_fold_cross_validation(df, None, k=5)

    return model