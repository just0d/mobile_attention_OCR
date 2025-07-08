import os
import random
import numpy as np
import cv2
import tensorflow as tf

from preprocess import enhanced_preprocess, label_to_sequence, alphabets, max_str_len, PAD_TOKEN

class AcademicDataGenerator(tf.keras.utils.Sequence):
    """Data generator for academic OCR training"""

    def __init__(self, df, batch_size=32, train_mode=True, augment=True):
        self.df = df
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.augment = augment and train_mode

        # Split indices into train or validation
        if train_mode:
            self.indices = list(range(0, int(len(df) * 0.8)))
        else:
            self.indices = list(range(int(len(df) * 0.8), len(df)))

        # Filter only valid samples
        self.valid_indices = []
        for idx in self.indices:
            try:
                row = self.df.iloc[idx]
                text = row['text']
                if (len(text) > 1 and len(text) <= max_str_len - 5 and
                    all(c in alphabets for c in text)):
                    self.valid_indices.append(idx)
            except:
                continue

        # Repeat data if too small to meet batch size * 3
        original_valid_count = len(self.valid_indices)
        if original_valid_count > 0:
            target_size = max(batch_size * 3, original_valid_count)
            while len(self.valid_indices) < target_size:
                add_count = min(original_valid_count, target_size - len(self.valid_indices))
                self.valid_indices.extend(self.valid_indices[:add_count])

    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Correct index bounds
        if end_idx > len(self.valid_indices):
            end_idx = len(self.valid_indices)
            start_idx = max(0, end_idx - self.batch_size)

        batch_indices = self.valid_indices[start_idx:end_idx]

        # Pad batch if smaller than batch_size
        while len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            batch_indices.extend(self.valid_indices[:remaining])

        batch_indices = batch_indices[:self.batch_size]

        return self._generate_batch(batch_indices)

    def _generate_batch(self, batch_indices):
        X = []
        decoder_input = []
        decoder_target = []

        for idx in batch_indices:
            try:
                row = self.df.iloc[idx]
                text = row['text']
                img_path = row['image_path']

                # Load and preprocess image
                if '/dummy/' in img_path:
                    processed_img = np.random.randn(64, 256, 1).astype(np.float32)
                else:
                    possible_paths = [
                        f"/content/drive/MyDrive/iam_data/{img_path}",
                        f"/content/drive/MyDrive/iam_data/data/{os.path.basename(img_path)}",
                        f"/content/drive/MyDrive/iam_data/words/{img_path}"
                    ]

                    loaded = False
                    for full_path in possible_paths:
                        if os.path.exists(full_path):
                            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                processed_img = enhanced_preprocess(img)
                                processed_img = processed_img.astype(np.float32) / 255.0
                                processed_img = np.expand_dims(processed_img, axis=-1)
                                # Normalize to [-1, 1]
                                processed_img = (processed_img - 0.5) / 0.5
                                loaded = True
                                break

                    if not loaded:
                        processed_img = np.random.randn(64, 256, 1).astype(np.float32)

                # Convert text to label sequence
                sequence = label_to_sequence(text)
                decoder_input_seq = sequence[:-1]
                decoder_target_seq = sequence[1:]

                # Pad sequences to max length
                while len(decoder_input_seq) < max_str_len:
                    decoder_input_seq.append(PAD_TOKEN)
                while len(decoder_target_seq) < max_str_len:
                    decoder_target_seq.append(PAD_TOKEN)

                X.append(processed_img)
                decoder_input.append(decoder_input_seq)
                decoder_target.append(decoder_target_seq)

            except Exception:
                # On failure, return empty batch element
                X.append(np.zeros((64, 256, 1), dtype=np.float32))
                decoder_input.append([PAD_TOKEN] * max_str_len)
                decoder_target.append([PAD_TOKEN] * max_str_len)

        X = np.array(X, dtype=np.float32)
        decoder_input = np.array(decoder_input, dtype=np.int32)
        decoder_target = tf.keras.utils.to_categorical(decoder_target, num_classes=len(alphabets) + 2)

        return [X, decoder_input], decoder_target

    def on_epoch_end(self):
        if self.train_mode:
            random.shuffle(self.valid_indices)