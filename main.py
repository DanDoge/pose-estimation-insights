import argparse
import time
import os
#from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import torch.optim as optim

from voc import VOCDetection, TransformVOCDetectionAnnotation
from pascal3d_plus import PasDet
from gen_map import generate_aps

import importlib

#from model import model

from tqdm import tqdm

import numpy as np

CUDA_AVAILABLE = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Faster R-CNN Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='model',
                    help='file containing model definition '
                    '(default: model)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
'''
cls = ('__background__', # always index 0
       'aeroplane', 'bicycle', 'bird', 'boat',
       'bottle', 'bus', 'car', 'cat', 'chair',
       'cow', 'diningtable', 'dog', 'horse',
       'motorbike', 'person', 'pottedplant',
       'sheep', 'sofa', 'train', 'tvmonitor')
'''
cls = ('__background__', # always index 0
        'aeroplane', 'bicycle', 'boat',
        'bottle', 'bus', 'car', 'chair',
        'diningtable',
        'motorbike',
        'sofa', 'train', 'tvmonitor')
class_to_ind = dict(zip(cls, range(len(cls))))

args = parser.parse_args()
model = importlib.import_module(args.model).model()
model_test = importlib.import_module(args.model).model()
model_test.load_state_dict(model.state_dict())


import pickle

objs = pickle.load(open("./data/objs_" + args.data + ".pkl", "rb"))
train_data = PasDet(objs, transform = transforms.ToTensor())
val_data = PasDet(objs, transform = transforms.ToTensor())

def collate_fn(batch):
    imgs, gt, id = zip(*batch)
    return imgs[0].unsqueeze(0), gt[0], id

train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=True,
            num_workers=0, collate_fn=collate_fn)


val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=collate_fn)

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)


def train(train_loader, model, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  #iou = AverageMeter()

  model.train()
  if CUDA_AVAILABLE:
      model.cuda()
  end = time.time()
  if epoch % 1 == 0:
      model.save_model(epoch)
  for i, (im, gt, id) in (enumerate(train_loader)):
    adjust_learning_rate(optimizer, epoch)

    # measure data loading time
    data_time.update(time.time() - end)

    optimizer.zero_grad()
    if CUDA_AVAILABLE:
      loss, scores, boxes, viewpoints = model((im.cuda(), gt))
    else:
      loss, scores, boxes, viewpoints = model((im, gt))
    #print(scores[gt["gt_classes"]], boxes.shape, id, gt)
    if loss is None:
        continue
    loss.backward()
    optimizer.step()

    #print(loss.data)

    losses.update(loss.data, im.size(0))
    #sum_iou, n_bbox = get_iou(scores, boxes, gt)
    #iou.update(sum_iou / n_bbox, n_bbox)
    #print('IOU {sum_iou:.4f} ({n_bbox:.4f})\t'.format(sum_iou=sum_iou, n_bbox=n_bbox))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            .format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses
            #top1=top1, top5=top5
            ))
      #global model_test
      #assert model.state_dict() == model_test.state_dict()

def validate(val_loader, model, epoch):
  batch_time = AverageMeter()
  losses = AverageMeter()

  # switch to evaluate mode
  model.eval()
  if CUDA_AVAILABLE:
      model.cuda()
  end = time.time()

  det = {}
  for i in range(12):
      det[str(i)] = []

  for i, (im, gt, id) in enumerate(val_loader):
    if CUDA_AVAILABLE:
      loss, scores, boxes, viewpoints = model((im.cuda(), gt))
    else:
      loss, scores, boxes, viewpoints = model((im, gt))
    if loss is None:
        continue
    losses.update(loss.data, im.size(0))
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    #print(viewpoints.shape, boxes.shape)

    if i % args.print_freq == 0:
      print('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            .format(
            i, len(val_loader), batch_time=batch_time,
            #data_time=data_time,
            loss=losses,
            #top1=top1, top5=top5
            ))

    for bbox_idx in range(scores.shape[0]):
      conf, idx = torch.max(scores[bbox_idx], 0)
      conf /= scores[bbox_idx].sum(0)
      #print(viewpoints[bbox_idx].view(13, 24).sum(0))
      _, viewpoint_pred = torch.max(viewpoints[bbox_idx].view(13, 24).sum(0), 0)
      if idx != 0:
        #print(str((idx - 1).item()))
        bbox = boxes[bbox_idx][idx * 4 : idx * 4 + 4]
        det[str((idx - 1).item())].append({"bbox":bbox, "conf":conf.item(), "id":id, "viewpoint":viewpoint_pred})
        #print(det)

    #if i > 10:
    #  break;

  #print(det)

  class_names = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
  for label in class_names:
      if os.path.exists("./res/" + label + ".txt"):
          os.remove("./res/" + label + ".txt")
  for label in range(12):
    for bbox in det[str(label)]:
      with open("./res/" + class_names[label] + ".txt", "a+") as fp:
        fp.write("{} {:3f} {} {}".format(bbox["id"][0].split('\'')[0], bbox["conf"], " ".join(["{:3f}".format(num.item()) for num in bbox["bbox"]]), bbox["viewpoint"]))
        fp.write("\n")

  generate_aps(epoch=epoch)




def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.state_dict()['param_groups']:
    param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

for epoch in range(0, 50):
  train(train_loader, model, optimizer, epoch)
  if epoch % 10 == 0:
      validate(val_loader, model, epoch)
  #model.save_model(1)
  #model.load_model("fastRCNN_1.pt")
  #generate_aps(epoch=epoch)

#from IPython import embed; embed()

#if __name__ == '__main__':
#  main()
