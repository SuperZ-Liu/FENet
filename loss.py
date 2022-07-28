"""
Loss.py
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, size_average=True,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.jointwtborder:
        if args.joint_edgeseg_loss:
            # joint edge loss with boundary relax
            criterion = JointEdgeSegLoss(classes=args.dataset_cls.num_classes,
                                         ignore_index=args.dataset_cls.ignore_label,
                                         edge_weight=args.edge_weight, seg_weight=args.seg_weight,
                                         att_weight=args.att_weight).cuda()
        else:
            # add ohem option
            criterion = ImgWtLossSoftNLL(classes=args.dataset_cls.num_classes,
                                     ignore_index=args.dataset_cls.ignore_label,
                                     upper_bound=args.wt_bound, ohem=args.ohem).cuda()
    elif args.fpn_dsn_loss:
        criterion = CriterionSFNet(ignore_index=args.dataset_cls.ignore_label)

    else:
        if args.ohem and args.aux:
            criterion = OhemWithAux(ignore_index=args.dataset_cls.ignore_label).cuda()
        elif args.ohem and not args.aux:
            criterion = OhemCrossEntropy2dTensor(ignore_index=args.dataset_cls.ignore_label).cuda()
        else:
            criterion = CrossEntropyLoss2d(size_average=True,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    criterion_val = CrossEntropyLoss2d(size_average=True,
                                       weight=None,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()
    return criterion, criterion_val


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )


class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False, ohem=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = False
        self.ohem = ohem
        self.ohem_loss = OhemCrossEntropy2dTensor(self.ignore_index).cuda()

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def onehot2label(self, target):
        # a bug here
        label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        label[target[:, -1, :, :]] = self.ignore_index
        return label

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            if self.ohem:
                return self.ohem_loss(inputs, self.onehot2label(target))
            border_weights = 1 / border_weights
            target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        # add ohem loss for the final stage
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH) and self.ohem:
            return self.ohem_loss(inputs, self.onehot2label(target[:,:-1,:,:]))
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1
        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss


class OhemWithAux(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000, aux_weight=0.4):
        super(OhemWithAux, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.aux_weight = aux_weight
        self.main_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)
        self.aux_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, pred, target):
        x_main, x_aux = pred
        return self.main_loss(x_main, target) + self.aux_weight * self.aux_loss(x_aux, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


def boundary_cross_entropy_loss2d(inputs, targets):
    # balance is decided by your dataset
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * 1. / valid
    weights = torch.Tensor(weights).cuda()
    loss = nn.BCELoss(weights)(inputs, targets)
    return loss


def cross_entropy2d_boundary(input, target):
    # balance is decided by your dataset
    n, c, h, w = input.size()
    weights = np.zeros(20)

    t_0 = (t == 0).sum()
    t_1 = (t == 1).sum()
    t_2 = (t == 2).sum()
    t_3 = (t == 3).sum()
    t_4 = (t == 4).sum()
    t_5 = (t == 5).sum()
    t_6 = (t == 6).sum()
    t_7 = (t == 7).sum()
    t_8 = (t == 8).sum()
    t_9 = (t == 9).sum()
    t_10 = (t == 10).sum()
    t_11 = (t == 11).sum()
    t_12 = (t == 12).sum()
    t_13 = (t == 13).sum()
    t_14 = (t == 14).sum()
    t_15 = (t == 15).sum()
    t_16 = (t == 16).sum()
    t_17 = (t == 17).sum()
    t_18 = (t == 18).sum()
    neg = (t == 255).sum() # unlabel & no_edge
    valid = neg + t_0 + t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10 + t_11 + t_12 + t_13 + t_14 + t_15 + t_16 + t_17 + t_18
    weights[i, t == 0] = t_0 * 1. / valid
    weights[i, t == 1] = t_1 * 1. / valid
    weights[i, t == 2] = t_2 * 1. / valid
    weights[i, t == 3] = t_3 * 1. / valid
    weights[i, t == 4] = t_4 * 1. / valid
    weights[i, t == 5] = t_5 * 1. / valid
    weights[i, t == 6] = t_6 * 1. / valid
    weights[i, t == 7] = t_7 * 1. / valid
    weights[i, t == 8] = t_8 * 1. / valid
    weights[i, t == 9] = t_9 * 1. / valid
    weights[i, t == 10] = t_10 * 1. / valid
    weights[i, t == 11] = t_11 * 1. / valid
    weights[i, t == 12] = t_12 * 1. / valid
    weights[i, t == 13] = t_13 * 1. / valid
    weights[i, t == 14] = t_14 * 1. / valid
    weights[i, t == 15] = t_15 * 1. / valid
    weights[i, t == 16] = t_16 * 1. / valid
    weights[i, t == 17] = t_17 * 1. / valid
    weights[i, t == 18] = t_18 * 1. / valid
    weights[i, t == 255] = neg * 1. / valid
    weights = torch.Tensor(weights).cuda()

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weights, size_average=False)
    return loss


def WeightedMultiLabelSigmoidLoss(model_output, target):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES
    转成one-hot
    """
    # Calculate weight. (edge pixel and non-edge pixel)
    weight_sum = utils.check_gpu(0, target.sum(dim=1).sum(dim=1).sum(dim=1).float().data)  # BS
    edge_weight = utils.check_gpu(0, weight_sum.data / float(target.size()[1] * target.size()[2]))
    non_edge_weight = utils.check_gpu(0, (target.size()[1] * target.size()[2] - weight_sum.data) / float(
        target.size()[1] * target.size()[2]))
    one_sigmoid_out = sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    target = target.transpose(1, 3).transpose(2, 3).float()  # BS X NUM_CLASSES X H X W
    loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * target * torch.log(
        one_sigmoid_out.clamp(min=1e-10)) - \
           edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * (1 - target) * torch.log(
        zero_sigmoid_out.clamp(min=1e-10))

    return loss.mean(dim=0).sum()


class CriterionSFNet(nn.Module):
    def __init__(self, aux_weight=1.0, thresh=0.7, min_kept=100000,  ignore_index=255):
        super(CriterionSFNet, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        seg_gt, boundary_gt = target
        h, w = seg_gt.size(1), seg_gt.size(2)
        main_pred, coarse_preds, boundary_pre = preds

        main_pred = F.interpolate(main_pred, size=(h, w), mode='bilinear', align_corners=True)
        main_loss = self._criterion1(main_pred, seg_gt)

        if len(coarse_preds) is not 0:
            for i in range(len(coarse_preds)):
                coarse_seg = F.interpolate(coarse_preds[i], size=(h, w), mode='bilinear', align_corners=True)
                main_loss += self._aux_weight * self._criterion1(coarse_seg, seg_gt)
        if len(boundary_pre) is not 0:
            boundary_pre = F.interpolate(boundary_pre[0], size=(h, w), mode='bilinear', align_corners=True)
            main_loss += boundary_cross_entropy_loss2d(boundary_pre, boundary_gt)
        return main_loss


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, ignore_index=255,mode='train',
                 edge_weight=1, seg_weight=1, seg_body_weight=1, att_weight=1):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index).cuda()
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(size_average=True,
                                               ignore_index=ignore_index).cuda()

        self.seg_body_loss = ImgWtLossSoftNLL(classes=classes,
                                     ignore_index=ignore_index,
                                     upper_bound=1.0, ohem=False).cuda()
        self.edge_ohem_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index, min_kept=5000).cuda()

        self.ignore_index = ignore_index
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.seg_body_weight = seg_body_weight


    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0


        weight = torch.from_numpy(weight).cuda()
        log_p = log_p.cuda()
        target_t = target_t.cuda()

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        filler = torch.ones_like(target) * 255
        return self.edge_ohem_loss(input, torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        seg_in, seg_body_in, edge_in = inputs
        seg_bord_mask, edgemask = targets
        segmask = self.onehot2label(seg_bord_mask)
        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(seg_in, segmask)
        losses['seg_body'] = self.seg_body_weight * self.seg_body_loss(seg_body_in, seg_bord_mask)
        losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edge_in, edgemask)
        losses['edge_ohem_loss'] = self.att_weight * self.edge_attention(seg_in, segmask, edge_in)

        return losses

    def onehot2label(self, target):
        """
        Args:
            target:

        Returns:

        """
        label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        label[target[:, -1, :, :]] = self.ignore_index
        return label