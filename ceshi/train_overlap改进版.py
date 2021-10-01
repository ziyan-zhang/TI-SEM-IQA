"""-----------------------------------------------------
创建时间 :  2020/6/14  19:53
说明    :
todo   :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from scipy import stats
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.optim as optim
from pathlib import Path
import math
import matplotlib.pyplot as plt

from Conv5_0_NoRelu512dp16_single import netReg
from DL_OL_XIN import train_dataset, val_dataset, data_name, cut_number


# from DL_128 import trainLoader, valLoader, dataName, batchSize, cutNumber

# from KangLe_CNNIQA import CNNIQAnet

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
    return ('%d小时%d分钟%.3f秒' % (hour_t, minite_t, t))

def rmse_function(a, b):
    """does not need to be array-like"""
    a = np.array(a)
    b = np.array(b)
    mse = ((a-b)**2).mean()
    rmse = math.sqrt(mse)
    return rmse

def get_perform(perf_dict):

    val_srcc_mean = perf_dict['val_srcc_mean']
    val_plcc_mean = perf_dict['val_plcc_mean']
    val_rmse_mean = perf_dict['val_rmse_mean']

    val_srcc_mean = np.array(val_srcc_mean)
    theIndex = np.argmax(val_srcc_mean)

    val_srcc_mean_max = val_srcc_mean[theIndex]
    val_plcc_mean_max = val_plcc_mean[theIndex]
    val_rmse_mean_max = val_rmse_mean[theIndex]

    return val_srcc_mean_max, val_plcc_mean_max, val_rmse_mean_max

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
        self.train_patience = 200

        self.plot_epoch = plot_epoch
        # 定义模型
        self.model = regmodel
        print('模型包含%d个参数' % sum([p.data.nelement() for p in self.model.parameters()]))
        
        self.adam_lr = LR
        print('使用的学习率是%.1f' % self.adam_lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', verbose=True, min_lr=1e-6, patience=100, factor=0.1, threshold=1e-4)

        # 指定文件夹, 和哪一个模型状态(最新/最佳)
        self.resume = False
        if self.resume == False:
            self.saveFolder = save_folder

            self.lr_bin = []
            # 定义一些收集数据的列表
            self.train_loss_bin = []
            self.train_srcc_bin = []
            self.train_plcc_bin = []

            self.val_loss_bin = []
            self.val_srcc_bin = []
            self.best_val_srcc = -1
            self.val_plcc_bin = []

            self.val_srcc_mean_bin = []
            self.val_plcc_mean_bin = []
            self.val_rmse_mean_bin = []

            self.is_best = True
            self.is_best_mean = True
        else:
            self.resumeFolder = ''
            self.saveFolder = self.resumeFolder

            model_path = 'epoch29_ckpt.pth.tar'
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
            self.val_srcc_bin = perf_dict['val_srcc']
            self.best_val_srcc = np.array(perf_dict['val_srcc']).max()
            self.val_plcc_bin = perf_dict['val_plcc']

            self.val_srcc_mean_bin = perf_dict['val_srcc_mean']
            self.val_plcc_mean_bin = perf_dict['val_plcc_mean']

            self.is_best = False
            self.is_best_mean = False

    def train(self):
        # 第一个epoch True, 第一个epoch之后可以改
        for epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            # train_loss, train_srcc, train_plcc, train_klcc, train_mse = self.train_one_epoch(epoch)
            train_loss, train_srcc, train_plcc = self.train_one_epoch()
            val_loss, val_srcc, val_plcc, val_srcc_mean, val_plcc_mean, val_rmse_mean = self.validate(epoch)

            self.scheduler.step(val_srcc_mean)
            temp_lr = self.optimizer.param_groups[0]['lr']
            print('当前学习率%.1e' % temp_lr)
            self.lr_bin.append(temp_lr)

            f = open('%s/lr_dict.txt' % self.saveFolder, 'w')
            f.write(str(self.lr_bin))
            f.close()

            # 打印当epoch的性能
            msg1 = 'Train>>>  srcc: {0:6.3f}  plcc: {1:6.3f}   VAL>>> srcc: {2:6.3f}  plcc: {3:6.3f}        BEST_VAL>>> srcc: {4:6.3f}  plcc: {5:6.3f}'
            msg2 = '                                  VAL_MEAN>>> srcc: {0:6.3f}  plcc: {1:6.3f}   BEST_VAL_MEAN>>> srcc: {2:6.3f}  plcc: {3:6.3f}'

            if epoch >= self.start_epoch + 1:
                self.is_best_mean = val_srcc_mean > np.array(self.val_srcc_mean_bin).max()
                self.is_best = val_srcc > np.array(self.val_srcc_bin).max()

            if self.is_best_mean:
                self.counter = 0
                self.best_val_srcc_mean = val_srcc_mean
                msg2 += '[^]'
            else:
                self.counter += 1
                print('已经%d个epoch平均SRCC没有提升' % self.counter)

            if self.is_best:
                self.best_val_srcc = val_srcc
                msg1 += '[*]'
            else:
                pass

            if self.counter > self.train_patience:
                val_srcc_mean_max, val_plcc_mean_max, val_rmse_mean_max = get_perform(perform_dict)
                print('>>>>>   SRCC: %.3f,  PLCC: %.3f,  RMSE: %.3f   <<<<<' % (
                    val_srcc_mean_max, val_plcc_mean_max, val_rmse_mean_max))
                print('[!]已经%d个epoch没有性能提升, 停止训练' % self.train_patience)
                return

            self.save_checkpoint(epoch,
                                 {'epoch': epoch+1,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  'best_val_srcc': self.best_val_srcc,
                                  'best_val_srcc_mean': self.best_val_srcc_mean},
                                 self.is_best_mean)

            # 记录相关数据
            self.train_loss_bin.append(train_loss)
            self.train_srcc_bin.append(train_srcc)
            self.train_plcc_bin.append(train_plcc)

            self.val_loss_bin.append(val_loss)
            self.val_srcc_bin.append(val_srcc)
            self.val_plcc_bin.append(val_plcc)
            self.val_srcc_mean_bin.append(val_srcc_mean)
            self.val_plcc_mean_bin.append(val_plcc_mean)
            self.val_rmse_mean_bin.append(val_rmse_mean)

            print(msg1.format(train_srcc, train_plcc, val_srcc, val_plcc, np.array(self.val_srcc_bin).max(), np.array(self.val_plcc_bin).max()))
            print(msg2.format(val_srcc_mean, val_plcc_mean, np.array(self.val_srcc_mean_bin).max(), np.array(self.val_plcc_mean_bin).max()))

            # 画图
            if epoch % self.plot_epoch == self.plot_epoch-1:
                plt.figure()
                plt.plot(self.train_loss_bin, label='train loss', linewidth=0.5)
                plt.plot(self.train_srcc_bin, label='train srcc')
                # plt.plot(self.train_plcc_bin, label='train plcc')

                plt.plot(self.val_loss_bin, label='val loss', linewidth=0.5)
                plt.plot(self.val_srcc_bin, label='val srcc')
                # plt.plot(self.val_plcc_bin, label='val plcc')

                plt.plot(self.val_srcc_mean_bin, label='val mean srcc')
                # plt.plot(self.val_plcc_mean_bin, label='val mean plcc')
                # plt.plot(self.val_rmse_mean_bin, label='val mean rmse')

                plt.plot(np.log10(self.lr_bin) * 0.27 + 1.6, label='lr: %.1e' % self.adam_lr, linestyle='--')

                max_val_srcc_mean = max(self.val_srcc_mean_bin)
                max_val_srcc_index = self.val_srcc_mean_bin.index(max_val_srcc_mean)
                plt.scatter(max_val_srcc_index, max_val_srcc_mean, marker='x')

                plt.legend()
                plt.ylim(0, 1.1)
                plt.title('srcc_mean_最后%.3f\nsrcc_mean_最大%.3f' % (self.val_srcc_mean_bin[-1], np.array(self.val_srcc_mean_bin).max()))
                plt.savefig(os.path.join(self.saveFolder, 'perform.png'))
                plt.close('all')

            perform_dict = {'train_loss': self.train_loss_bin, 'train_srcc': self.train_srcc_bin, 'train_plcc': self.train_plcc_bin,
                            'val_loss': self.val_loss_bin, 'val_srcc': self.val_srcc_bin, 'val_plcc': self.val_plcc_bin,
                            'val_srcc_mean': self.val_srcc_mean_bin, 'val_plcc_mean': self.val_plcc_mean_bin, 'val_rmse_mean': self.val_rmse_mean_bin}

            f = open('%s/perform_dict.txt' % self.saveFolder, 'w')
            f.write(str(perform_dict))
            f.close()

    def train_one_epoch(self):
        self.model.train()
        loss_bin = AverageMeter()
        batch_time = AverageMeter()
        predict_bin = []
        y_bin = []

        tic = time.time()
        with tqdm(total=len(self.train_loader.sampler)) as pbar:
            # 更改变2-1: 输入加载器
            for i, (imgRead, y) in enumerate(self.train_loader):
                if self.use_gpu:
                    imgRead= imgRead.cuda()
                    y = y.cuda()
                y = y.float()  # todo: 看能不能省去

                self.batch_size = imgRead.shape[0]

                # 前向传播
                predict = self.model(imgRead)
                predict = predict.squeeze()

                # 损失函数, 以及反向传播, 更新梯度
                loss = F.mse_loss(predict, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_bin.update(loss.cpu().item())

                # 收集数据
                predict_bin.extend(predict.detach().cpu().numpy())
                y_bin.extend(y.cpu().numpy())

                # 计算用的时间
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description('{:.1f}s - loss: {:.3f}'.format((toc-tic), loss.item()))
                pbar.update(self.batch_size)

        srcc = stats.spearmanr(predict_bin, y_bin)[0]
        plcc = stats.pearsonr(predict_bin, y_bin)[0]

        return loss_bin.avg, srcc, plcc

    def validate(self, epoch):
        self.model.eval()
        loss_bin = AverageMeter()
        batch_time = AverageMeter()
        predict_bin = []
        y_bin = []

        tic = time.time()
        # todo: 内存增加的话加上个with torch.no_grad()
        with torch.no_grad():
            with tqdm(total=len(self.val_loader.sampler)) as pbar:
                for i, (imgRead, y) in enumerate(self.val_loader):
                    if self.use_gpu:
                        imgRead= imgRead.cuda()
                        y = y.cuda()
                    y = y.float()

                    self.batch_size = imgRead.shape[0]

                    # 更改变3-3: 前向传播模型输入改动-测试. 加上r17
                    predict = self.model(imgRead)
                    predict = predict.squeeze()

                    # 损失函数, 不再反向传播和更新梯度
                    loss = F.mse_loss(predict, y)
                    loss_bin.update(loss.item())

                    # 收集数据
                    predict_bin.extend(predict.detach().cpu().numpy())
                    y_bin.extend(y.cpu().numpy())

                    # 计算时间
                    toc = time.time()
                    batch_time.update(toc-tic)

                    pbar.set_description('{:.1f}s - loss: {:.3f}'.format((toc-tic), loss.item()))
                    pbar.update(self.batch_size)

        predict_bin = np.array(predict_bin)
        y_bin = np.array(y_bin)

        srcc_raw = stats.spearmanr(predict_bin, y_bin)[0]
        plcc_raw = stats.pearsonr(predict_bin, y_bin)[0]

        # 求平均值
        predict_bin_rs = predict_bin.reshape(-1, self.cutNumber)
        y_bin_rs = y_bin.reshape(-1, self.cutNumber)

        predict_mean = predict_bin_rs.mean(axis=1)
        y_mean = y_bin_rs.mean(axis=1)
        assert len(y_mean) == 195
        # 求CC
        srcc_mean = stats.spearmanr(predict_mean, y_mean)[0]
        plcc_mean = stats.pearsonr(predict_mean, y_mean)[0]
        rmse_mean = rmse_function(predict_mean, y_mean)

        return loss_bin.avg, srcc_raw, plcc_raw, srcc_mean, plcc_mean, rmse_mean

    def save_checkpoint(self, epoch, state, is_best_mean):
        filename = 'epoch' + str(epoch) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.saveFolder, filename)
        torch.save(state, ckpt_path)

        filename_minus1 = 'epoch' + str(epoch-1) + '_ckpt.pth.tar'
        ckpt_path_minus1 = os.path.join(self.saveFolder, filename_minus1)

        if os.path.exists(ckpt_path_minus1):
            os.remove(ckpt_path_minus1)

        if is_best_mean:
            filename = 'model_BEST.pth.tar'
            copyfile(ckpt_path, os.path.join(self.saveFolder, filename))


def main():  # 单独使用main函数是为了避免使用全局变量, 是接口更严谨
    regmodel = netReg()
    # regmodel = CNNIQAnet()

    regmodel.cuda()
    ModelName = regmodel.model_name

    JiaZai = True
    if JiaZai:
        ckptdir = '0616_0825'
        ckptname = 'netG_latest.pth'
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
    batchSize = 200
    LR = 5e-4
    num_train_samples = 16*10

    # 结果命名： 程序+模型+数据+LRBS+时间, 执行在生成训练加载器和测试加载器之前，瞬间完成，防止记录错误
    train_file_name = Path(__file__).name

    saveFolder = train_file_name[:-3] + ModelName + '_%s_%d' % (dataName, num_train_samples) + \
                 '_LR%.1e_bs%d' % (LR, batchSize) + '_T' + time.strftime('%m%d_%H%M%S')
    saveFolder = 'ceshi'

    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    # 训练代码记录一下，为了复现。更牛逼的工作还要记录随机种子。
    model_file_name = 'Conv5_0_NoRelu512dp16_single.py'
    data_file_name = 'DL_OL_XIN.py'
    copyfile(train_file_name, os.path.join(saveFolder, train_file_name))
    copyfile(model_file_name, os.path.join(saveFolder, model_file_name))
    copyfile(data_file_name, os.path.join(saveFolder, data_file_name))
    print('训练，模型，数据的代码已备份。')

    trainDataset = train_dataset()
    if num_train_samples == 7280:
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, pin_memory=False)
    elif num_train_samples < 7280:
        train_sampler = sampler.SubsetRandomSampler(np.random.choice(range(len(trainDataset)), num_train_samples))
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, sampler=train_sampler)
    else:
        raise ValueError('训练的样本数不对')

    valDataset = val_dataset()
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, pin_memory=False)
    CUTNUMBER = cut_number()
    PLOT_EPOCH = 5
    
    trainer = Trainer(trainLoader, valLoader, regmodel, saveFolder, LR, CUTNUMBER, PLOT_EPOCH)
    trainer.train()

if __name__ == '__main__':
    global_tic = time.time()
    main()
    global_toc = time.time()
    print('用时%s' % char_shijian(global_toc-global_tic))


