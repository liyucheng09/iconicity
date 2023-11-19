import datasets

if __name__ == '__main__':
    imgaenet = datasets.load_dataset('imagenet-1k', split='train')
    