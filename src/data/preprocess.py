import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

def one_hot_to_label(row):
    for col in ['healthy', 'multiple_diseases', 'rust', 'scab']:
        if row[col] == 1:
            return col
    return 'unknown'

def main(input_csv, input_dir, output_dir, img_size, val_split, random_seed):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    df['label'] = df.apply(one_hot_to_label, axis=1)

    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=random_seed)

    for subset, subset_df in [('train', train_df), ('val', val_df)]:
        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for _, row in subset_df.iterrows():
            label_dir = os.path.join(subset_dir, row['label'])
            os.makedirs(label_dir, exist_ok=True)
            img_path = os.path.join(input_dir, row['image_id'] + '.jpg')
            dst_path = os.path.join(label_dir, row['image_id'] + '.jpg')
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))
            img.save(dst_path)

    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input_csv, args.input_dir, args.output_dir, args.img_size, args.val_split, args.random_seed)
