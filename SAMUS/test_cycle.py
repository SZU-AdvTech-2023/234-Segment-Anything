from ast import arg
import os
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
from thop import profile
from tqdm import tqdm
import io

def main():
    #  =========================================== parameters setting ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str,help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256,help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128,help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='CAMUS', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b',help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='/data/gjx/project/dataset/checkpoint/sam_vit_b_01ec64.pth',help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1,help='batch_size per gpu')  # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001,help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')  # 0.0006
    parser.add_argument('--warmup', type=bool, default=True,help='If activated, warp up the learning from a lower lr to the base_lr')  # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--gpu_id', type=str, default='0')
    # parser.add_argument('--weight_folder', type = str, default='/data/gjx/project/SAMUS/SAMUS/checkpoints/CAMUS/vit_b_50epochs', help = 'List of weight files to test')
    args = parser.parse_args()


    opt = get_config(args.task)  # please configure your hyper-parameter
    print("task", args.task)
    opt.mode = "val"
    # opt.classes=2
    opt.visual = True
    # opt.eval_mode = "patient"
    opt.modelname = args.modelname
    device = torch.device(opt.device)

    # ===========================================================分布式训练===================================================================================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================

    # register the sam model
    opt.batch_size = args.batch_size * args.n_gpu
    # opt.batch_size = args.batch_size

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    val_dataset = ImageToImage2D(opt.data_path, tf_val, mode='test')  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.modelname == "SAMed":
        opt.classes = 2
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()
#===========================================================================================================================
    weight_folder = '/data/gjx/project/SAMUS/SAMUS/checkpoints/CAMUS/SAMUS_desam'
    weight_files = sorted([os.path.join(weight_folder, file) for file in os.listdir(weight_folder) if file.endswith('.pth')])
    # for file in sorted(os.listdir(weight_folder)):
    #     print(os.path.join(weight_folder,file))


    # print("dataset:" + args.task + " -----------model name: " + args.modelname)
    #  ========================================================================= begin to evaluate the model ============================================================================

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total_params: {}".format(pytorch_total_params))
    # input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    # points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    # flops, params = profile(model, inputs=(input, points), )
    # # 模型的参数数量: params 和浮点运算次数: Gflops
    # print('Gflops:', flops / 1000000000, 'params:', params)
#=============================================================================
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    criterion = get_criterion(modelname=args.modelname, opt=opt)
    #==============================cycle test=============================
    for epoch,weight_file in enumerate(tqdm(weight_files, desc="test_cycle:")):
        # with open(weight_file, 'rb') as f:
        #     buffer = io.BytesIO(f.read())
        print("file:",weight_file)
        checkpoint = torch.load(weight_file)
        # ------when the load model is saved under multiple GPU
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)


        model.eval()
        # print("mode",opt.mode)     mode=val
        if opt.mode == "train":
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print("mean dice:", mean_dice)
        else:
            # mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            dice, hd = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            # print("task", args.task, "checkpoints:", weight_file)
            print(f"epoch[{epoch+1}/{opt.epochs}]:\nDice: {dice}\nHD: {hd}")
        with open("/data/gjx/project/SAMUS/SAMUS/tensorboard/test_cycle/SAMUS_desam.txt","a+") as file:
            file.write(f"[{epoch+1}/{opt.epochs}]:\n")
            file.write(f"Dice:{dice}\tHD:{hd}\n")
            file.close()
        # print(mean_dice[1:], mean_hdis[1:], mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
        # print(std_dice[1:], std_hdis[1:], std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])

        # with open("experiments.txt", "a+") as file:
        #     file.write(args.task + " " + args.modelname + "-pt10 " + '%.2f'%(mean_dice[1]) + "±" + '%.2f'%std_dice[1] + " ")
        #     file.write('%.2f'%mean_hdis[1] + "±" + '%.2f'%std_hdis[1] + " ")
        #     file.write('%.2f'%(mean_iou[1]) + "±" + '%.2f'%std_iou[1] + " ")
        #     file.write('%.2f'%(mean_acc[1]) + "±" + '%.2f'%std_acc[1] + " ")
        #     file.write('%.2f'%(mean_se[1]) + "±" + '%.2f'%std_se[1] + " ")
        #     file.write('%.2f'%(mean_sp[1]) + "±" + '%.2f'%std_sp[1] + "\n")

        #     # file.write("MYO " + args.modelname + " " + '%.2f'%(mean_dice[2]) + "±" + '%.2f'%std_dice[2] + " ")
        #     # file.write('%.2f'%mean_hdis[2] + "±" + '%.2f'%std_hdis[2] + " ")
        #     # file.write('%.2f'%(mean_iou[2]) + "±" + '%.2f'%std_iou[2] + " ")
        #     # file.write('%.2f'%(mean_acc[2]) + "±" + '%.2f'%std_acc[2] + " ")
        #     # file.write('%.2f'%(mean_se[2]) + "±" + '%.2f'%std_se[2] + " ")
        #     # file.write('%.2f'%(mean_sp[2]) + "±" + '%.2f'%std_sp[2] + "\n")
        #     # file.write("LA " + args.modelname + " " + '%.2f'%(mean_dice[3]) + "±" + '%.2f'%std_dice[3] + " ")
        #     # file.write('%.2f'%mean_hdis[3] + "±" + '%.2f'%std_hdis[3] + " ")
        #     # file.write('%.2f'%(mean_iou[3]) + "±" + '%.2f'%std_iou[3] + " ")
        #     # file.write('%.2f'%(mean_acc[3]) + "±" + '%.2f'%std_acc[3] + " ")
        #     # file.write('%.2f'%(mean_se[3]) + "±" + '%.2f'%std_se[3] + " ")
        #     # file.write('%.2f'%(mean_sp[3]) + "±" + '%.2f'%std_sp[3] + "\n")
        #     file.close()


if __name__ == '__main__':
    main()
