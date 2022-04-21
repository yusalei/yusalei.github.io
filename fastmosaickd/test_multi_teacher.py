import registry
import torch
import argparse
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(current_lr, val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        print(' [Eval] Epoch={current_epoch} Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f} Loss={losses.avg:.4f} Lr={lr:.4f}'
                .format(current_epoch=0, top1=top1, top5=top5, losses=losses, lr=current_lr))
    return top1.avg


parser = argparse.ArgumentParser(description='MosaicKD for OOD data')
parser.add_argument('--teacher', default='resnet34')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--data_root', default='data')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
args = parser.parse_args()
num_classes = 10

teacher_1 = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
teacher_1.load_state_dict(torch.load('checkpoints/pretrained/cifar10-resnet34_8x-0.pt', map_location='cpu'))
teacher_2 = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
teacher_2.load_state_dict(torch.load('checkpoints/pretrained/cifar10-resnet34_8x-1.pt', map_location='cpu'))
teacher_3 = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
teacher_3.load_state_dict(torch.load('checkpoints/pretrained/cifar10-resnet34_8x-2.pt', map_location='cpu'))
teacher_4 = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
teacher_4.load_state_dict(torch.load('checkpoints/pretrained/cifar10-resnet34_8x-3.pt', map_location='cpu'))
teacher_5 = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
teacher_5.load_state_dict(torch.load('checkpoints/pretrained/cifar10-resnet34_8x-4.pt', map_location='cpu'))
teacher_1 = teacher_1.cuda()
teacher_2 = teacher_2.cuda()
teacher_3 = teacher_3.cuda()
teacher_4 = teacher_4.cuda()
teacher_5 = teacher_5.cuda()
teacher = registry.EnsembleModel([teacher_1, teacher_2, teacher_3, teacher_4, teacher_5]).cuda()
criterion = nn.CrossEntropyLoss().cuda()

num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
acc1 = validate(0, val_loader, teacher, criterion, args)

