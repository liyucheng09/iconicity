import datasets
from PIL import Image
from classes import IMAGENET2012_CLASSES
from tqdm import tqdm

IMAGENET2012_CLASSES = list(IMAGENET2012_CLASSES.values())

if __name__ == '__main__':
    imagenet = datasets.load_dataset('imagenet-1k.py', split='validation')
    count = 0
    labels = []
    images = []
    for i in tqdm(imagenet):
        image = i['image']
        label = i['label']
        if label not in labels:
            textual_label = IMAGENET2012_CLASSES[label]
            labels.append(label)
            images.append(image)
            textual_label = textual_label.replace(' ', '_').replace(',', '')

            image.save(f'images/{textual_label}-{label}.jpg', 'JPEG')