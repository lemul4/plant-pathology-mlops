import argparse
import os
from PIL import Image
import numpy as np
import albumentations as A

def main(input_dir, output_dir, n_aug, img_size, horizontal_flip, rotate_limit, brightness_contrast):
    os.makedirs(output_dir, exist_ok=True)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
        A.Rotate(limit=rotate_limit, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=brightness_contrast,
                                   contrast_limit=brightness_contrast, p=0.5),
        A.Resize(img_size, img_size)
    ])

    for subset in ['train', 'val']:
        in_subset = os.path.join(input_dir, subset)
        out_subset = os.path.join(output_dir, subset)
        if not os.path.isdir(in_subset):
            continue
        for class_name in os.listdir(in_subset):
            in_class = os.path.join(in_subset, class_name)
            out_class = os.path.join(out_subset, class_name)
            os.makedirs(out_class, exist_ok=True)
            for fname in os.listdir(in_class):
                src_path = os.path.join(in_class, fname)
                img = np.array(Image.open(src_path).convert('RGB'))
                Image.fromarray(img).save(os.path.join(out_class, fname))
                if subset == 'train':
                    for i in range(n_aug):
                        aug = transform(image=img)['image']
                        dst_path = os.path.join(out_class, f"{fname.split('.')[0]}_aug{i}.jpg")
                        Image.fromarray(aug).save(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_aug", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--horizontal_flip", type=bool, default=True)
    parser.add_argument("--rotate_limit", type=int, default=20)
    parser.add_argument("--brightness_contrast", type=float, default=0.2)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.n_aug, args.img_size,
         args.horizontal_flip, args.rotate_limit, args.brightness_contrast)
