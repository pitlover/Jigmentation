import json
import numpy as np


def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent='\t')


class compute_miou:
    def __init__(self, json_path):
        data = read_json(json_path)
        self.class_names = ['background'] + data['class_names']
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask < 255
        correct_mask = (pred_mask == gt_mask) * obj_mask

        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(np.sum((pred_mask == i) * obj_mask))
            T_list.append(np.sum((gt_mask == i) * obj_mask))
            TP_list.append(np.sum((gt_mask == i) * correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask < 255
        correct_mask = (pred_mask == gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += np.sum((pred_mask == i) * obj_mask)
            self.T[i] += np.sum((gt_mask == i) * obj_mask)
            self.TP[i] += np.sum((gt_mask == i) * correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = []  # over activation
        FN_list = []  # under activation

        for i in range(self.classes):
            IoU = self.TP[i] / (self.T[i] + self.P[i] - self.TP[i] + 1e-10) * 100
            FP = (self.P[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)

        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))

        if clear:
            self.clear()

        if detail:
            return dict(mIoU=mIoU, mIoU_foreground=mIoU_foreground, IoU_dic=IoU_dic, FP=FP, FN=FN)
        else:
            return dict(mIoU=mIoU, mIoU_foreground=mIoU_foreground)

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []

        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)
