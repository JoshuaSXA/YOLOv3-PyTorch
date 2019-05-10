import torch
import torch.nn as nn
import numpy as np
import math


class YOLOLoss(nn.Module):
    def __init__(self, anchors, classes_num, img_size):
        super(YOLOLoss, self).__init__()
        self._anchors = anchors
        self._anchor_num = len(anchors)
        self._classes_num = classes_num
        self._img_size = img_size

        self._ignore_th = 0.5
        self._lambda_xy = 2.5
        self._lambda_wh = 2.5
        self._lambda_conf = 1.0
        self._lambda_cls = 1.0

        self._mse_loss = nn.MSELoss()
        self._bce_loss = nn.BCELoss()


    def forward(self, input, target=None):
        # batch size, height and width of the input maps
        batch_size = input.size(0)
        input_h = input.size(2)
        input_w = input.size(3)
        # stride of the whole network
        stride_h = self._img_size[1] / input_h
        stride_w = self._img_size[0] / input_w
        # calculate the anchor scale in the output maps of the network
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_height, anchor_width in self._anchors]
        # reshape the input maps
        prediction = input.view(batch_size, self._anchor_num, self._classes_num + 5, input_h, input_w).permute(0, 1, 3, 4, 2).contiguous()
        # extract x, y, h, w, conf and class prob
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        cls_prob = torch.sigmoid(prediction[..., 5:])

        if target is not None:
            # get the parsed target
            mask, void_mask, target_x, target_y, target_w, target_h, target_conf, target_cls = self.parse_target(target, scaled_anchors, input_w, input_h, self._ignore_th)
            if torch.cuda.is_available():
                # use GPU
                mask, void_mask = mask.cuda(), void_mask.cuda()
                target_x, target_y, target_w, target_h = target_x.cuda(), target_y.cuda(), target_w.cuda(), target_h.cuda()
                target_conf, target_cls = target_conf.cuda(), target_cls.cuda()
            n_mask = torch.sum(mask)
            n_void_mask = torch.sum(void_mask)

            # losses.
            loss_x = self._bce_loss(x * mask, target_x * mask) / n_mask
            loss_y = self._bce_loss(y * mask, target_y * mask) / n_mask
            loss_w = self._mse_loss(w * mask, target_w * mask) / n_mask
            loss_h = self._mse_loss(h * mask, target_h * mask) / n_mask
            loss_conf = self._bce_loss(conf * mask, 1.0 * mask) / n_mask + 0.5 * self._bce_loss(conf * void_mask, void_mask * 0.0) / n_void_mask
            loss_cls = self._bce_loss(cls_prob[mask == 1], target_cls[mask == 1]) / n_mask
            # total loss
            loss = (loss_x + loss_y) * self._lambda_xy + (loss_w + loss_h) * self._lambda_wh + loss_conf * self._lambda_conf + loss_cls * self._lambda_cls
            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, input_w - 1, input_w).repeat(input_w, 1).repeat(batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_h - 1, input_h).repeat(input_h, 1).t().repeat(batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_h * input_w).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_h * input_w).view(h.shape)
            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale, conf.view(batch_size, -1, 1), cls_prob.view(batch_size, -1, self.num_classes)), -1)
            return output.data


    def parse_target(self, target, anchors, input_w, input_h, ignore_th):
        batch_size = len(target)
        # allocate tensors to store the parsed results
        mask = torch.zeros(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        void_mask = torch.ones(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        target_x = torch.zeros(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        target_y = torch.zeros(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        target_w = torch.zeros(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        target_h = torch.zeros(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        target_conf = torch.zeros(batch_size, self._anchor_num, input_h, input_w, requires_grad=False)
        target_cls = torch.zeros(batch_size, self._anchor_num, input_h, input_w, self._classes_num, requires_grad=False)
        for batch in range(batch_size):
            for obj in range(target[batch].size(0)):
                if target[batch][obj].sum() == 0:
                    continue
                # the position relative to box
                gt_x = target[batch][obj, 1] * input_w
                gt_y = target[batch][obj, 2] * input_h
                gt_w = target[batch][obj, 3] * input_w
                gt_h = target[batch][obj, 4] * input_h

                # get grid box indices
                gt_i = int(gt_x)
                gt_j = int(gt_y)
                # get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gt_w, gt_h])).unsqueeze(0)
                # get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self._anchor_num, 2)), np.array(anchors)), axis=1))
                # calculate iou between gt and anchor shapes
                anchor_ious = self.bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                void_mask[batch, anchor_ious > ignore_th, gt_j, gt_i] = 0
                # Find the best matching anchor box
                best_anchors = np.argmax(anchor_ious)

                # Masks
                if (gt_j < input_h) and (gt_i < input_w):
                    mask[batch, best_anchors, gt_j, gt_i] = 1
                    # Coordinates
                    target_x[batch, best_anchors, gt_j, gt_i] = gt_x - gt_i
                    target_y[batch, best_anchors, gt_j, gt_i] = gt_y - gt_j
                    # Width and height
                    target_w[batch, best_anchors, gt_j, gt_i] = math.log(gt_w / anchors[best_anchors][0] + 1e-16)
                    target_h[batch, best_anchors, gt_j, gt_i] = math.log(gt_h / anchors[best_anchors][1] + 1e-16)
                    # object
                    target_conf[batch, best_anchors, gt_j, gt_i] = 1
                    # One-hot encoding of label
                    target_cls[batch, best_anchors, gt_j, gt_i, int(target[batch][obj, 0])] = 1
                else:
                    print('Step {0} out of bound'.format(batch))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gt_j, input_h, gt_i, input_w))
                    continue
        return mask, void_mask, target_x, target_y, target_w, target_h, target_conf, target_cls


    # IoU
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou