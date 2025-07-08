import os
import glob
import pandas as pd

def create_line_level_mapping():
    """Create line-level mapping between images and text from IAM dataset"""

    # Scan all image files in IAM data folder
    image_files = glob.glob('/content/drive/MyDrive/iam_data/data/*/*.png')

    # Map filename (without extension) to relative image path
    image_dict = {}
    for img_path in image_files:
        filename = os.path.basename(img_path).replace('.png', '')
        relative_path = img_path.replace('/content/drive/MyDrive/iam_data/', '')
        image_dict[filename] = relative_path

    words_txt_path = '/content/drive/MyDrive/iam_data/words.txt'

    # Group words by line ID from words.txt
    line_texts = {}

    with open(words_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.split()
            # Valid lines have at least 9 parts and second part 'ok'
            if len(parts) < 9 or parts[1] != 'ok':
                continue

            word_id = parts[0]
            text = parts[8].rstrip('\n')

            if not text or text in ['err', '###']:
                continue

            # Extract line ID from word ID (e.g. a01-000u-00-00 → a01-000u)
            line_id = '-'.join(word_id.split('-')[:2])

            if line_id not in line_texts:
                line_texts[line_id] = []
            line_texts[line_id].append(text.upper())

    # Build dataset list: image path + concatenated line text
    dataset = []
    matched = 0

    for line_id, words in line_texts.items():
        if line_id in image_dict:
            line_text = ' '.join(words)
            dataset.append({
                'image_path': image_dict[line_id],
                'text': line_text,
                'line_id': line_id,
                'filename': f"{line_id}.png",
                'word_count': len(words)
            })
            matched += 1

    return dataset

def create_final_iam_csv():
    """Create the final CSV for IAM dataset with correct file paths"""

    dataset = create_line_level_mapping()

    if len(dataset) == 0:
        print("No data found for IAM dataset.")
        return None

    df = pd.DataFrame(dataset)

    # Filter out empty text and lines with zero words
    df = df[df['text'].str.len() > 0]
    df = df[df['word_count'] >= 1]

    # Fix image paths to actual folder structure
    def fix_image_path(line_id):
        possible_folders = ['000', '001', '002', '003', '004', '005']

        for folder in possible_folders:
            test_path = f"data/{folder}/{line_id}.png"
            full_test_path = f"/content/drive/MyDrive/iam_data/{test_path}"
            if os.path.exists(full_test_path):
                return test_path

        # Fallback default path
        return f"data/000/{line_id}.png"

    df['image_path'] = df['line_id'].apply(fix_image_path)

    csv_path = '/content/drive/MyDrive/iam_dataset_IMPROVED.csv'
    df.to_csv(csv_path, index=False)
    print(f"Final IAM CSV saved to {csv_path}")

    return df