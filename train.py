import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=8, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if iter_num % 100 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'trained_model/model_final.pt')


if __name__ == '__main__':
    main()
