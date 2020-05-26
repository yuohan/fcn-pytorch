import argparse
from PIL import Image
import numpy as np
import torch

from models import FCN8

def load_image_tensor(image_path):

    image = Image.open(image_path)
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float()

def color_map(num_classes):

    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        n = i
        r = g = b = 0
        for j in range(8):
            r = r | (((n >> 0) & 1) << (7 - j))
            g = g | (((n >> 1) & 1) << (7 - j))
            b = b | (((n >> 2) & 1) << (7 - j))
            n >>= 3
        cmap[i] = np.array([r, g, b])
    return cmap

def make_label_image(pred, palette):

    image = Image.fromarray(pred.argmax(axis=0).cpu().numpy().astype(np.uint8))
    image.mode = 'P'
    image.putpalette(palette)
    return image

def main(image_path, model_path, use_cuda):

    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    model = FCN8.load(model_path, device)
    model.eval()

    image = load_image_tensor(image_path)
    with torch.no_grad():
        pred = model(image.unsqueeze(0))

    palette = color_map(256).reshape(-1)
    image = make_label_image(pred[0], palette)

    image.save('result.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation a Image with pretrained model')
    parser.add_argument('image', type=str,
                    help='Path of image to be segmented')
    parser.add_argument('--model-path', type=str, default='models/checkpoint.pth',
                    help='Saved model path')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()
    main(args.image, args.model_path, args.use_cuda)