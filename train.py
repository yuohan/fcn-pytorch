import argparse
from tqdm.auto import tqdm
import torch
from torch import optim, nn
from metrics import Metric
from datasets import VOC2012Dataset
from models import FCN8

def train_epoch(train_loader, model, criterion, optimizer, epoch, device):

    model.train()

    train_loss = 0
    for image, label in tqdm(train_loader, total=len(train_loader)):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label) / len(image)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def validate_epoch(val_loader, model, criterion, epoch, device):

    model.eval()

    val_loss = 0
    metric = Metric(['accuracy', 'mean_iou'], len(val_loader.dataset.classes))
    for image, label in tqdm(val_loader, total=len(val_loader)):
        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(image)
        loss = criterion(pred, label) / len(image)

        val_loss += loss.item()
        metric.update(pred.data.cpu().numpy(), label.data.cpu().numpy())
        
    metrics = metric.compute()
    return val_loss / len(val_loader), metrics['accuracy'], metrics['mean_iou']

def train_loop(train_loader, val_loader, model, criterion, optimizer, epochs, device):

    for epoch in range(1, epochs+1):

        print (f'epoch {epoch}/{epochs}')
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, device)
        val_loss, acc, mean_iou = validate_epoch(val_loader, model, criterion, epoch, device)

        print (f'val_loss: {val_loss:.4f} val_acc: {acc:.4f} val_mean_iou: {mean_iou:.4f}')
        print ('-'*10)

def main(root_dir, save_path, epochs, learning_rate, momentum, weight_decay, use_cuda):

    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

    # dataset
    train_dataset = VOC2012Dataset(root_dir, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = VOC2012Dataset(root_dir, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # models
    num_classes = len(train_dataset.classes)
    model = FCN8(num_classes).to(device)

    # trainer
    ignore_index = train_dataset.ignore_index

    optimizer = optim.SGD(model.parameters(), learning_rate, momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    train_loop(train_loader, val_loader, model, criterion, optimizer, epochs, device)

    if save_path is not None:
        model.save(save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train FCN Model for Image Segmentation')

    parser.add_argument('--root-dir', type=str, default='input/VOCdevkit/VOC2012',
                    help='Root directory of datasets')
    parser.add_argument('--epochs', type=int, default=5,
                    help='Number of Epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-10,
                    help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.99,
                    help='Momentum of SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Weight decay of SGD optimizer')
    parser.add_argument('--save-path', type=str, default='models/checkpoint.pth',
                    help='Model save path')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()
    main(args.root_dir, args.save_path,
        args.epochs, args.learning_rate, args.momentum, args.weight_decay, args.use_cuda)