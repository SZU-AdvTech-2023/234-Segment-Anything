from ast import arg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from tqdm import tqdm


def main():
    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str,help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256,help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128,help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='CAMUS', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b',help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str,default='/data/gjx/project/dataset/checkpoint/sam_vit_b_01ec64.pth',help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1,help='batch_size per gpu')  # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=6, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001,help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')  # 0.0006
    parser.add_argument('--warmup', type=bool, default=True,help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250,help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--gpu_id', type=str, default='0,1,3,5,6,7')
    args = parser.parse_args()  # 解析命令行参数，将其存储在args变量中
    opt = get_config(args.task)  # get_config 函数

    # device = torch.device(opt.device)
    if args.keep_log:  # 保留训练期间loss、ls、dice的值
        logtimestr = time.strftime('%m.%d-%H:%M')  # eg:11.17-12:12  initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================

    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size

    # opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None,long_mask=True)  # image reprocessing
    # tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, tf_train, mode='train')
    # val_dataset = ImageToImage2D(opt.data_path, tf_val, mode='val')  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    # valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)  # 加载模型
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr,betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9,weight_decay=0.0001)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs + 1), np.zeros(opt.epochs + 1)
    for epoch in range(opt.epochs):

        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        trainloader_iter = tqdm(enumerate(trainloader), desc=f'Training Epoch {epoch + 1}/{opt.epochs}', unit='batch',leave=False)
        for batch_idx, (datapack) in trainloader_iter:
            # initial_memory = torch.cuda.memory_allocated()
            # print("占用：", initial_memory)
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)
            # bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            # -------------------------------------------------------- forward --------------------------------------------------------
            imgs = imgs.squeeze(0)
            masks = masks.squeeze(0)
            #
            name = datapack['patient_name']
            # print("name",name)
            # print("imgs",imgs.shape)     # b c h w
            # print("masks",masks.shape)    # b c h w

            pred = model(imgs, pt)
            # print("pred",pred.size)
            # print(f"imgs{imgs.shape}\nmasks{masks.shape}")
            train_loss = criterion(pred, masks)
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
#=========================================更新进度条，显示train_loss==========================================
            trainloader_iter.set_postfix({'train_loss':train_loss})
            trainloader_iter.update()
            # print(train_loss)
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                del imgs, masks, pred, train_loss
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (
                                1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                    del imgs, masks, pred, train_loss
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch + 1, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)
            with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt','w') as f:
                for i in range(len(loss_log)):
                    f.write(str(loss_log[i]) + '\n')
        #  --------------------------------------------------------- keep pth ----------------------------------------------------------
        timestr = time.strftime('%m.%d-%H:%M')
        save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch + 1)
        if not os.path.isdir(opt.save_path):
            os.makedirs(opt.save_path)
        torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()