from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst
import torchvision.datasets as datasets

from utils import AverageMeter, attention_map, accuracy
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet
from SoftTarget import SoftTarget
import ResNet as ResNet

parser = argparse.ArgumentParser(description='train kd')

parser.add_argument('--save_root', type=str, default='./results/imagenet/', help='results path')
parser.add_argument('--img_root', type=str, default='./datasets', help='dataset path')
parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='ours_imagenet', help='file name')
parser.add_argument('--data_name', type=str, default='imagenet', help='name of dataset')
parser.add_argument('--lambda_kd', type=float, default=0.1, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature')
parser.add_argument("--adaptive_sum", default="ADAPTIVE", type=str, choices=['AVERAGE', 'ADAPTIVE'])

args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		cudnn.enabled = True
		cudnn.benchmark = True
	logging.info("args = %s", args)
	logging.info("unparsed_args = %s", unparsed)

	tnet1 = ResNet.resnet50(pretrained=True)
	tnet2 = ResNet.resnet34(pretrained=True)
	snet = ResNet.resnet18(pretrained=False)

	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()

	criterionKD = SoftTarget(args.T)

	optimizer = torch.optim.SGD(snet.parameters(),
								lr = args.lr,
								momentum = args.momentum,
								weight_decay = args.weight_decay,
								nesterov = True)

	traindir = os.path.join(args.img_root, 'train')
	valdir = os.path.join(args.img_root, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

	train_dataset = datasets.ImageFolder(traindir,
										 transforms.Compose([
										 transforms.RandomResizedCrop(224),
										 transforms.RandomHorizontalFlip(),
										 transforms.ToTensor(),
										 normalize])
									 )

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
										   num_workers=args.workers, pin_memory=True, sampler=None)

	val_dataset = datasets.ImageFolder(valdir,
								   transforms.Compose([
									   transforms.Resize(256),
									   transforms.CenterCrop(224),
									   transforms.ToTensor(),
									   normalize])
								   )
	test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
										 num_workers=args.workers, pin_memory=True)



	nets = {'snet':snet, 'tnet1':tnet1, 'tnet2':tnet2}
	criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD}

	best_top1 = 0
	best_top5 = 0
	for epoch in range(1, args.epochs+1):
		adjust_lr(optimizer, epoch)

		epoch_start_time = time.time()
		train(train_loader, nets, optimizer, criterions, epoch)

		logging.info('Testing the models......')
		test_top1, test_top5 = test(test_loader, nets, criterions, epoch)

		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		# save model
		is_best = False
		if test_top1 > best_top1:
			best_top1 = test_top1
			best_top5 = test_top5
			is_best = True
		logging.info('Saving models......')
		save_checkpoint({
			'epoch': epoch,
			'snet': snet.state_dict(),
			'tnet1': tnet1.state_dict(),
			'tnet2': tnet2.state_dict(),
			'prec@1': test_top1,
			'prec@5': test_top5,
			'best@1': best_top1,
			'best@5': best_top5,
		}, is_best, args.save_root)

def train(train_loader, nets, optimizer, criterions, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet1 = nets['tnet1']
	tnet2 = nets['tnet2']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.train()

	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		out_s = snet(img)
		out_t1 = tnet1(img)
		out_t2 = tnet2(img)
		out_t_list = []
		out_t_list.append(out_t1, out_t2)


		# 分类损失
		cls_loss = criterionCls(out_s[3], target)
		kd_loss_list = []
		if args.adaptive_sum == "AVERAGE":
			kd_loss1 = criterionKD(out_s[3], out_t1[3].detach())
			kd_loss2 = criterionKD(out_s[3], out_t2[3].detach())
			kd_loss_list.append(kd_loss1)
			kd_loss_list.append(kd_loss2)
			kd_loss = torch.stack(kd_loss_list).mean(0)
		elif args.ensemble_method == "ADAPTIVE":
			kd_loss1 = criterionKD(out_s[3], out_t1[3].detach())
			kd_loss2 = criterionKD(out_s[3], out_t2[3].detach())
			kd_loss_list.append(kd_loss1)
			kd_loss_list.append(kd_loss2)
			entropy_list = []
			softmax_out_t1 = F.softmax(out_t1[3], dim=1)
			entropy1 = -(softmax_out_t1) * (torch.log2(softmax_out_t1))
			entropy_list.append(entropy1.sum(1))
			softmax_out_t2 = F.softmax(out_t2[3], dim=1)
			entropy2 = -(softmax_out_t2) * (torch.log2(softmax_out_t2))
			entropy_list.append(entropy2.sum(1))
			entropy_sum = torch.stack(entropy_list, dim=0).sum(dim=0).squeeze(dim=0)
			w1 = 1.0 - (entropy_list[0] / entropy_sum)
			w2 = 1.0 - (entropy_list[1] / entropy_sum)



		soft_loss = args.alpha * cls_loss + (1 - args.alpha) * kd_loss

		s_attention1 = attention_map(out_s[0])
		s_attention2 = attention_map(out_s[1])
		s_attention3 = attention_map(out_s[2])
		t_attention1 = attention_map(out_t1[0])
		t_attention2 = attention_map(out_t1[1])
		t_attention3 = attention_map(out_t1[2])
		attention_loss_list = []
		attention_loss_list.append(F.l1_loss(s_attention1, t_attention1))
		attention_loss_list.append(F.l1_loss(s_attention2, t_attention2))
		attention_loss_list.append(F.l1_loss(s_attention3, t_attention3))
		attention_loss = torch.stack(kd_loss_list).mean(0)

		loss = soft_loss + args.beta * attention_loss

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
					   'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
			logging.info(log_str)


def test(test_loader, nets, criterions, epoch):
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.eval()
	if args.kd_mode in ['vid', 'ofd']:
		for i in range(1,4):
			criterionKD[i].eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		if args.kd_mode in ['sobolev', 'lwm']:
			img.requires_grad = True
			stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
			stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)
		else:
			with torch.no_grad():
				out_s = snet(img)
				out_t = tnet(img)

		cls_loss = criterionCls(out_s, target)
		if args.kd_mode in ['logits', 'st']:
			kd_loss  = criterionKD(out_s, out_t.detach()) * args.lambda_kd
		elif args.kd_mode in ['fitnet', 'nst']:
			kd_loss = criterionKD(rb3_s[1], rb3_t[1].detach()) * args.lambda_kd
		elif args.kd_mode in ['at', 'sp']:
			kd_loss = (criterionKD(rb1_s[1], rb1_t[1].detach()) +
					   criterionKD(rb2_s[1], rb2_t[1].detach()) +
					   criterionKD(rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['pkt', 'rkd', 'cc']:
			kd_loss = criterionKD(feat_s, feat_t.detach()) * args.lambda_kd
		elif args.kd_mode in ['fsp']:
			kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
					   criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
					   criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['ab']:
			kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
					   criterionKD(rb2_s[0], rb2_t[0].detach()) +
					   criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['sobolev']:
			kd_loss = criterionKD(out_s, out_t, img, target) * args.lambda_kd
		elif args.kd_mode in ['lwm']:
			kd_loss = criterionKD(out_s, rb2_s[1], out_t, rb2_t[1], target) * args.lambda_kd
		elif args.kd_mode in ['irg']:
			kd_loss = criterionKD([rb2_s[1], rb3_s[1], feat_s, out_s],
								  [rb2_t[1].detach(),
								   rb3_t[1].detach(),
								   feat_t.detach(),
								   out_t.detach()]) * args.lambda_kd
		elif args.kd_mode in ['vid', 'afd']:
			kd_loss = (criterionKD[1](rb1_s[1], rb1_t[1].detach()) +
					   criterionKD[2](rb2_s[1], rb2_t[1].detach()) +
					   criterionKD[3](rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['ofd']:
			kd_loss = (criterionKD[1](rb1_s[0], rb1_t[0].detach()) +
					   criterionKD[2](rb2_s[0], rb2_t[0].detach()) +
					   criterionKD[3](rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
		else:
			raise Exception('Invalid kd mode...')

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
	logging.info('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

	return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
	scale   = 0.1
	lr_list = [args.lr*scale] * 30
	lr_list += [args.lr*scale*scale] * 10
	lr_list += [args.lr*scale*scale*scale] * 10

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()