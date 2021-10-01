"""-----------------------------------------------------
创建时间 :  2020/6/14  19:53
说明    :
todo   : TODO:重大bug, 算SRCC的时候try一下，防止程序出错中断
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 必须在`import torch`语句之前设置才能生效
import torch
device = torch.device('cuda')
# device = torch.device('cpu')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
import numpy as np
from progressbar import *
from shutil import copyfile

import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
import math
from torch.utils.data import DataLoader

from net_Boose_IQA import netReg
from DL_OL_XIN import train_dataset, val_dataset, data_name, cut_number


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
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

def char_shijian(t):
    if t >= 3600:
        hour_t = int(t / 3600)
    else:
        hour_t = 0
    t = t - hour_t * 3600
    if t >= 60:
        minite_t = int(t / 60)
    else:
        minite_t = 0
    t = t - minite_t * 60
    return ('%d小时%d分钟%.4f秒' % (hour_t, minite_t, t))

def rmse_function(a, b):
    """does not need to be array-like"""
    a = np.array(a)
    b = np.array(b)
    mse = ((a-b)**2).mean()
    rmse = math.sqrt(mse)
    return rmse

def get_perform(perf_dict):

    val_srcc = perf_dict['val_srcc_mean']
    val_plcc = perf_dict['val_plcc_mean']

    val_srcc = np.array(val_srcc)
    theIndex = np.argmax(val_srcc)

    val_srcc_max = val_srcc[theIndex]
    val_plcc_max = val_plcc[theIndex]

    return val_srcc_max, val_plcc_max

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Trainer(object):
    def __init__(self, data_loader_train, data_loader_val, regmodel, save_folder, LR, CUTNUMBER, plot_epoch):

        self.train_loader = data_loader_train
        self.val_loader = data_loader_val
        self.cutNumber = CUTNUMBER

        # 训练相关的参数
        self.epochs = 2000
        self.start_epoch = 0
        self.use_gpu = True
        self.counter = 0
        self.train_patience = 15
        self.plot_epoch = plot_epoch


        # 定义模型
        self.model = regmodel
        print('模型包含%d个参数' % sum([p.data.nelement() for p in self.model.parameters()]))

        self.adam_lr = LR
        print('使用的学习率为%.0e' % self.adam_lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_lr)
        # self.model = torch.nn.DataParallel(self.model)  # 就在这里wrap一下，模型就会使用所有的GPU

        # 指定文件夹, 和哪一个模型状态(最新/最佳)
        self.resume = False
        if self.resume == False:
            self.saveFolder = save_folder

            self.lr_bin = []
            self.counter_bin = []
            # 定义一些收集数据的列表
            self.train_loss_bin = []
            self.train_srcc_bin = []
            self.train_plcc_bin = []

            self.val_loss_bin = []
            self.val_srcc_mean_bin = []
            self.val_plcc_mean_bin = []

            self.is_best_srcc_mean = True
            self.is_best_loss = True
        else:
            self.resumeFolder = ''
            self.saveFolder = self.resumeFolder

            model_path = 'e'
            self.ckpt_path = os.path.join(self.resumeFolder, model_path)
            print('恢复上次训练, 从%s中提取模型' % self.ckpt_path)

            # 加载模型
            ckpt = torch.load(self.ckpt_path)
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optim_state'])

            # 加载性能字典
            dictPath = os.path.join(self.resumeFolder, 'perform_dict.txt')
            lrPath = os.path.join(self.resumeFolder, 'lr_dict.txt')

            with open(dictPath, 'r') as f:
                perf = f.read()
            perf_dict = eval(perf)

            with open(lrPath, 'r') as f:
                lr = f.read()
            lr_list = eval(lr)

            self.lr_bin = lr_list

            self.train_loss_bin = perf_dict['train_loss']
            self.train_srcc_bin = perf_dict['train_srcc']
            self.train_plcc_bin = perf_dict['train_plcc']

            self.val_loss_bin = perf_dict['val_loss']
            self.val_srcc_mean_bin = perf_dict['val_srcc_mean']
            self.val_plcc_mean_bin = perf_dict['val_plcc_mean']

            self.is_best_loss = False
            self.is_best_srcc_mean = False

    def train(self):
        # 第一个epoch True, 第一个epoch之后可以改
        for epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            # train_loss, train_srcc, train_plcc, train_klcc, train_mse = self.train_one_epoch(epoch)
            train_loss, train_srcc, train_plcc = self.train_one_epoch()
            val_loss, val_srcc_mean, val_plcc_mean = self.validate(epoch)

            temp_lr = self.optimizer.param_groups[0]['lr']
            print('当前学习率%.1e' % temp_lr)
            self.lr_bin.append(temp_lr)
            self.counter_bin.append(self.counter)

            f = open('%s/lr_dict.txt' % self.saveFolder, 'w')
            f.write(str(self.lr_bin))
            f.close()

            # 打印当epoch的性能
            print('train_loss: %.3f, val_loss: %.3f' % (train_loss, val_loss))
            msg1 = 'Train>>> srcc: {0:6.4f} plcc: {1:6.4f}  VAL_MEAN>>> srcc: {2:6.4f} plcc: {3:6.4f}  BEST_VAL_mean>>> srcc: {4:6.4f}  plcc: {5:6.4f}'

            if epoch >= self.start_epoch + 1:
                self.is_best_loss = (train_loss < np.array(self.train_loss_bin).min())
                self.is_best_srcc_mean = (val_srcc_mean > np.array(self.val_srcc_mean_bin).max())

            if self.is_best_loss:
                self.counter = 0
                msg1 += '[^]'
            else:
                self.counter += 1
                print('已经%d个epoch loss没有下降' % self.counter)

            self.save_checkpoint(epoch,
                                 {'epoch': epoch+1,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict()},
                                 self.is_best_srcc_mean)

            # 记录相关数据
            self.train_loss_bin.append(train_loss)
            self.train_srcc_bin.append(train_srcc)
            self.train_plcc_bin.append(train_plcc)

            self.val_loss_bin.append(val_loss)
            self.val_srcc_mean_bin.append(val_srcc_mean)
            self.val_plcc_mean_bin.append(val_plcc_mean)

            print(msg1.format(train_srcc, train_plcc, val_srcc_mean, val_plcc_mean, np.array(self.val_srcc_mean_bin).max(), np.array(self.val_plcc_mean_bin).max()))

            # 画图
            if epoch % self.plot_epoch == self.plot_epoch-1:
                plt.figure()
                plt.plot(self.train_loss_bin, label='train loss', linewidth=0.5)
                plt.plot(self.train_srcc_bin, label='train srcc')
                # plt.plot(self.train_plcc_bin, label='train plcc')

                plt.plot(self.val_loss_bin, label='val loss', linewidth=0.5)
                plt.plot(self.val_srcc_mean_bin, label='val mean srcc')
                # plt.plot(self.val_plcc_mean_bin, label='val mean plcc')

                plt.plot(np.log10(self.lr_bin) * 0.27 + 1.6, label='lr: %.1e' % self.adam_lr, linestyle='--')
                plt.plot(np.array(self.counter_bin)/100, label='counter', linewidth=0.5)

                max_val_srcc_mean = max(self.val_srcc_mean_bin)
                max_val_srcc_index = self.val_srcc_mean_bin.index(max_val_srcc_mean)
                plt.scatter(max_val_srcc_index, max_val_srcc_mean, marker='x')
                plt.plot([max_val_srcc_mean]*len(self.val_srcc_mean_bin), linewidth=0.5, linestyle='--')

                plt.legend()
                plt.ylim(0, 1.1)
                plt.title('srcc_mean_最后%.4f\nsrcc_mean_最大%.4f' % (self.val_srcc_mean_bin[-1], np.array(self.val_srcc_mean_bin).max()))
                plt.savefig(os.path.join(self.saveFolder, 'perform.png'))
                plt.close('all')

            perform_dict = {'train_loss': self.train_loss_bin, 'train_srcc': self.train_srcc_bin, 'train_plcc': self.train_plcc_bin,
                            'val_loss': self.val_loss_bin, 'val_srcc_mean': self.val_srcc_mean_bin, 'val_plcc_mean': self.val_plcc_mean_bin,
                            'lr': self.lr_bin, 'counter': self.counter_bin}

            f = open('%s/perform_dict.txt' % self.saveFolder, 'w')
            f.write(str(perform_dict))
            f.close()

            if self.counter > self.train_patience:
                if temp_lr > self.adam_lr * 0.01001:
                    adjust_learning_rate(self.optimizer, temp_lr*0.1)
                    self.counter = 0
                    print('这里自适应改变学习率...')
                else:
                    val_srcc_mean_max, val_plcc_mean_max = get_perform(perform_dict)
                    print('>>>>>   SRCC: %.4f,  PLCC: %.4f    <<<<<' % (
                        val_srcc_mean_max, val_plcc_mean_max))
                    print('[!]已经%d个epoch没有性能提升, 停止训练' % self.train_patience)

                    savefolderBig = (self.saveFolder).split('/')[0]
                    savefolderSmall = (self.saveFolder).split('/')[1]
                    newsavefolderSmall = 'srcc_%.4f_' % val_srcc_mean_max + savefolderSmall
                    os.rename(self.saveFolder, os.path.join(savefolderBig, newsavefolderSmall))
                    return

    def train_one_epoch(self):
        self.model.train()
        loss_bin = AverageMeter()
        batch_time = AverageMeter()
        q_predict_bin = []
        y_mean_bin = []

        tic = time.time()
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Counter(), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        progress = ProgressBar(widgets=widgets)

        # 更改变2-1: 输入加载器
        for i, (imgRead, y) in enumerate(progress(self.train_loader)):
            if self.use_gpu:
                imgRead= imgRead.to(device)
                y = y.to(device)
            y = y.float()  # todo: 看能不能省去

            self.batch_size = imgRead.shape[0]

            # 前向传播
            y_pred, alpha = self.model(imgRead)
            y_pred = y_pred.squeeze()
            alpha = alpha.squeeze()

            weighted_pred = torch.mul(y_pred, alpha)

            weighted_pred_rs = weighted_pred.reshape(-1, self.cutNumber)
            alpha_rs = alpha.reshape(-1, self.cutNumber)
            y_rs = y.reshape(-1, self.cutNumber)

            weight_pred_mean = weighted_pred_rs.mean(dim=1)  # todo: 按理说应该改成.sum,不过经过除法对结果应该没有任何影响
            alpha_mean = alpha_rs.mean(dim=1)
            y_mean = y_rs.mean(dim=1)

            q_pred = torch.div(weight_pred_mean, alpha_mean)

            loss_imagewise = F.l1_loss(q_pred, y_mean)
            loss_patchwise = F.l1_loss(y_pred, y)

            loss = loss_imagewise + loss_patchwise

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_bin.update(loss.cpu().item())

            # 收集数据
            q_predict_bin.extend(q_pred.detach().cpu().numpy())
            y_mean_bin.extend(y_mean.cpu().numpy())

            # 计算用的时间
            toc = time.time()
            batch_time.update(toc-tic)

        srcc = stats.spearmanr(q_predict_bin, y_mean_bin)[0]
        plcc = stats.pearsonr(q_predict_bin, y_mean_bin)[0]

        return loss_bin.avg, srcc, plcc

    def validate(self, epoch):
        self.model.eval()
        loss_bin = AverageMeter()
        batch_time = AverageMeter()
        q_predict_bin = []
        y_mean_bin = []

        tic = time.time()
        # todo: 内存增加的话加上个with torch.no_grad()
        with torch.no_grad():
            widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Counter(), ' ', Timer(),
                       ' ', ETA(), ' ', FileTransferSpeed()]
            progress = ProgressBar(widgets=widgets)
            for i, (imgRead, y) in enumerate(progress(self.val_loader)):
                if self.use_gpu:
                    imgRead= imgRead.to(device)
                    y = y.to(device)
                y = y.float()

                self.batch_size = imgRead.shape[0]

                # 前向传播
                y_pred, alpha = self.model(imgRead)
                y_pred = y_pred.squeeze()
                alpha = alpha.squeeze()

                weighted_pred = torch.mul(y_pred, alpha)

                weighted_pred_rs = weighted_pred.reshape(-1, self.cutNumber)
                alpha_rs = alpha.reshape(-1, self.cutNumber)
                y_rs = y.reshape(-1, self.cutNumber)

                weight_pred_mean = weighted_pred_rs.mean(dim=1)
                alpha_mean = alpha_rs.mean(dim=1)
                y_mean = y_rs.mean(dim=1)

                q_pred = torch.div(weight_pred_mean, alpha_mean)

                loss_imagewise = F.l1_loss(q_pred, y_mean)
                loss_patchwise = F.l1_loss(y_pred, y)

                loss = loss_imagewise + loss_patchwise

                loss_bin.update(loss.cpu().item())

                # 收集数据
                q_predict_bin.extend(q_pred.detach().cpu().numpy())
                y_mean_bin.extend(y_mean.cpu().numpy())

                # 计算用的时间
                toc = time.time()
                batch_time.update(toc - tic)

        srcc = stats.spearmanr(q_predict_bin, y_mean_bin)[0]
        plcc = stats.pearsonr(q_predict_bin, y_mean_bin)[0]

        return loss_bin.avg, srcc, plcc

    # def save_checkpoint(self, epoch, state, is_best_srcc_meean):
    #     filename = 'epoch' + str(epoch) + '_ckpt.pth.tar'
    #     ckpt_path = os.path.join(self.saveFolder, filename)
    #     torch.save(state, ckpt_path)
    #
    #     filename_minus1 = 'epoch' + str(epoch-1) + '_ckpt.pth.tar'
    #     ckpt_path_minus1 = os.path.join(self.saveFolder, filename_minus1)
    #
    #     if os.path.exists(ckpt_path_minus1):
    #         os.remove(ckpt_path_minus1)
    #
    #     if is_best_srcc_mean:
    #         filename = 'model_BEST.pth.tar'
    #         shutil.copyfile(ckpt_path, os.path.join(self.saveFolder, filename))

    def save_checkpoint(self, epoch, state, is_best_srcc_mean):
        if is_best_srcc_mean:
            filename = 'model_BEST_SRCC.pth.tar'
            ckpt_path = os.path.join(self.saveFolder, filename)
            torch.save(state, ckpt_path)


def main():  # 单独使用main函数是为了避免使用全局变量, 是接口更严谨
    regmodel = netReg()
    ModelName = regmodel.model_name
    regmodel.cuda()
    # regmodel = torch.nn.DataParallel(regmodel)
    # regmodel = CNNIQAnet()


    # JiaZai = True
    JiaZai = False
    if JiaZai:
        ckptdir = '0616_0825'
        ckptname = 'netG_latest.pth'
        # ckptdir = 'new_ce'
        # ckptname = 'netG_latest_3300.pth'
        ckptPath = os.path.join(ckptdir, ckptname)
        ModelName += ckptdir + ckptname

        ce_dict = torch.load(ckptPath)['state_dict']
        print('加载原来的权重%s' % ModelName)
        reg_dict = regmodel.state_dict()
        pretrained_dict = {k: v for k, v in ce_dict.items() if k in reg_dict}
        assert len(pretrained_dict) > 0, '模型不吻合！'
        reg_dict.update(pretrained_dict)
        regmodel.load_state_dict(reg_dict)
    else:
        print('从头开始训练')

    dataName = data_name()
    batchSize = 64
    LR = 5e-4
    num_train_samples = 7280

    train_file_name = Path(__file__).name

    bigFolder = time.strftime('D_%m%d')
    if not os.path.exists(bigFolder):
        os.mkdir(bigFolder)
    saveFolder = bigFolder + '/' + train_file_name[:-3] + ModelName + '_%s_%d' % (dataName, num_train_samples) + \
                 '_LR%.1e_bs%d' % (LR, batchSize) + '_T' + time.strftime('%m%d_%H%M%S')
    # saveFolder = 'ceshi'

    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    # 训练代码记录一下，为了复现。更牛逼的工作还要记录随机种子。
    data_file_name = 'DL_OL_XIN.py'
    model_boose_name = 'net_Boose_IQA.py'
    copyfile(train_file_name, os.path.join(saveFolder, train_file_name))
    copyfile(model_boose_name, os.path.join(saveFolder, model_boose_name))
    copyfile(data_file_name, os.path.join(saveFolder, data_file_name))
    print('训练，模型，数据的代码已备份。')



    trainDataset = train_dataset()
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False, pin_memory=True)

    valDataset = val_dataset()
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, pin_memory=True)
    CUTNUMBER = cut_number()
    PLOT_EPOCH = 5

    trainer = Trainer(trainLoader, valLoader, regmodel, saveFolder, LR, CUTNUMBER, PLOT_EPOCH)
    trainer.train()


if __name__ == '__main__':
    global_tic = time.time()
    main()
    global_toc = time.time()
    print('该次训练用时%s' % char_shijian(global_toc-global_tic))
