'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

#import torch.nn as nn
#import torch.nn.init as init

from xlrd import open_workbook
from xlutils.copy import copy
import xlwt


def time_write_excel(value_list,excel_file, next_row = True):

    if os.path.exists(excel_file):
        rexcel = open_workbook(excel_file)
        cols = rexcel.sheets()[0].ncols
        excel = copy(rexcel)
        table = excel.get_sheet(0)
    else:
        excel = xlwt.Workbook()
        table = excel.add_sheet('time used')
        cols = 0
    col = cols
    if next_row:
        for i in range(len(value_list)):
            table.write(i+1, col-1, value_list[i])
    else:
        for i in range(len(value_list)):
            table.write(i, col, value_list[i])
    excel.save(excel_file)

def result_write_excel(value_list,excel_file):
    if os.path.exists(excel_file):
        rexcel = open_workbook(excel_file)
        rows = rexcel.sheets()[0].nrows
        excel = copy(rexcel)
        table = excel.get_sheet(0)
    else:
        excel = xlwt.Workbook()
        table = excel.add_sheet('Experiment data')
        rows = 0
    row = rows
    for i in range(len(value_list)):
        table.write(row, i, value_list[i])
    excel.save(excel_file)


class Recall_And_Precison(object):
    """Computes and stores the average recall and precison rate"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.TP = 0.
        self.FN = 0
        self.FP = 0
        self.sum_recall = 0
        self.sum_precison = 0
        self.count = 0
        self.avg_recall = 0
        self.avg_precison = 0
        #self.cls_dict = {'handup':0, 'listen':1, 'negative':2, 'positive':3, 'write':4 }    #my
        self.cls_dict = {'listen':0, 'write':1, 'handup':2, 'positive':3, 'negative':4}  # wenqaing

    def update(self, pose_cls, predicted, targets, n = 1):

        self.TP += ((predicted == self.cls_dict[pose_cls]) & (targets.data == self.cls_dict[pose_cls])).cpu().sum()
        self.FN += ((predicted != self.cls_dict[pose_cls]) & (targets.data == self.cls_dict[pose_cls])).cpu().sum()
        self.FP += ((predicted == self.cls_dict[pose_cls]) & (targets.data != self.cls_dict[pose_cls])).cpu().sum()
        self.sum_recall = self.TP + self.FN
        self.sum_precison = self.TP + self.FP
        self.count += n
        if self.TP == 0:
            self.avg_recall = 0.
            self.avg_precison = 0.
        else:
            self.avg_recall = self.TP / self.sum_recall
            self.avg_precison = self.TP / self.sum_precison


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 60.  #65
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append(' Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    '''
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    '''
    sys.stdout.write('\r')
    sys.stdout.write('\t\t\t')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
