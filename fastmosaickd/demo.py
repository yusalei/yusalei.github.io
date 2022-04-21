import datafree
import torch
import argparse
import torch.nn as nn
from datafree.models.generator import Generator


def prepare_model(model):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        return model
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            return model
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            return model
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        return model
    else:
        model = torch.nn.DataParallel(model).cuda()
        return model

def reset_l0(model):
    for n,m in model.named_modules():
        print(n)
        if n == 'l1.0' or n == 'conv_blocks.0':
        #     nn.init.normal_(m.weight, 0.0, 0.02)
        #     nn.init.constant_(m.bias, 0)

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')
args = parser.parse_args()
args.distributed = False
args.gpu = 0

nz = 100
generator = Generator(nz=nz, ngf=64, img_size=32, nc=3)
generator = prepare_model(generator)
generator = generator.train()
reset_l0(generator)
