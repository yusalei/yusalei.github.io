import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
import time
import engine

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or  n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class FastMetaSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, discriminator, real_loader, nz, num_classes, img_size,           # todo discriminator, real_loader
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False, lr_z = 0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                 is_maml=1, local=1.0, align=1.0, balance=1.0):
        super(FastMetaSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.ismaml = is_maml
        self.local = local
        self.align = align
        self.balance = balance

        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        # todo
        self.discriminator = discriminator.to(device).train()           # todo train() 在这里写了
        # todo
        self.ood_loader = DataIter(real_loader)
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
            # todo
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
            # todo
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.lr_g * self.iterations,
                                                betas=[0.5, 0.999])

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m, self.bn_mmt) )
        self.aug = transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])

    def synthesize(self, targets=None):

        start = time.time()

        self.ep+=1
        self.student.eval()
        self.teacher.eval()         # todo ensemble teacher eval的方式
        best_cost = 1e6

        if (self.ep == 120+self.ep_start) and self.reset_l0:
            reset_l0(self.generator)
            reset_l0(self.discriminator)                # todo 不确定是不是要这么做
        
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        fast_generator = self.generator.clone()             # todo attention
        fast_discriminator = self.discriminator.clone()     # todo

        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        # todo
        optimizer_d = torch.optim.Adam(fast_discriminator.parameters(), lr=self.lr_g, betas=[0.5, 0.999])

        # todo real_loader->ood data to train discriminator
        for it in range(self.iterations):
            real = self.ood_loader.next()[0]
            real = real.cuda(self.device, non_blocking=True)

            ###############################
            # Patch Discrimination
            ###############################
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs)
            images = self.normalizer(inputs_aug)
            d_out_fake = fast_discriminator(images.detach())
            d_out_real = fast_discriminator(real.detach())
            loss_d = (torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake),
                                                                            reduction='sum') + \
                     torch.nn.functional.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real),
                                                                               reduction='sum')) / (
                                     2 * len(d_out_fake)) * self.local
            optimizer_d.zero_grad()                 # discriminator
            loss_d.backward()

            if self.ismaml:  # ismaml = 1
                if it == 0: self.d_optimizer.zero_grad()  # {generator}
                fomaml_grad(self.discriminator, fast_discriminator)
                if it == (self.iterations - 1): self.d_optimizer.step()

            optimizer_d.step()

            ###############################
            # Generation / Inversion Loss
            ###############################
            t_out = self.teacher(images)
            s_out = self.student(images)

            pyx = torch.nn.functional.softmax(t_out, dim=1)  # p(y|G(z))
            log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
            py = pyx.mean(0)  # p(y)

            # Mosaicking to distill
            d_out_fake = fast_discriminator(images)
            # (Eqn. 3) fool the patch discriminator
            loss_local = torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake,
                                                                            torch.ones_like(d_out_fake),
                                                                            reduction='sum') / len(d_out_fake)
            # (Eqn. 4) label space aligning
            loss_align = -(pyx * log_softmax_pyx).sum(1).mean()
            # (Eqn. 7) fool the student
            loss_adv = - engine.criterions.kldiv(s_out, t_out)

            # Appendix: Alleviating Mode Collapse for unconditional GAN
            loss_balance = (py * torch.log2(py)).sum()

            # Final loss: L_align + L_local + L_adv (DRO) + L_balance
            loss_g = self.adv * loss_adv + loss_align * self.align + self.local * loss_local + loss_balance * self.balance

            with torch.no_grad():
                if best_cost > loss_g.item() or best_inputs is None:
                    best_cost = loss_g.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()                   # fast_generator + z
            loss_g.backward()

            if self.ismaml:  # ismaml = 1
                if it == 0:
                    self.meta_optimizer.zero_grad()  # {generator}
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()

            optimizer.step()

        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            reptile_grad(self.discriminator, fast_discriminator)
            self.meta_optimizer.step()
            self.d_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add(best_inputs)
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}, end - start

    def sample(self):
        return self.data_iter.next()
