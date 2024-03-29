import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import numpy.random as npr

from utils import \
    bbox_transform, bbox_transform_inv, clip_boxes, bbox_overlaps

from utils import to_var as _tovar

CUDA_AVAILABLE = torch.cuda.is_available()

# should handle multiple scales, how?
class FasterRCNN(nn.Container):

  def __init__(self,
          features, pooler,
          classifier, rpn,
          batch_size=128, fg_fraction=0.25,
          fg_threshold=0.5, bg_threshold=None,
          num_classes=13):
    super(FasterRCNN, self).__init__()
    self.features = features
    self.roi_pooling = pooler
    self.rpn = rpn
    self.classifier = classifier

    self.batch_size = batch_size
    self.fg_fraction = fg_fraction
    self.fg_threshold = fg_threshold
    if bg_threshold is None:
        bg_threshold = (0, 0.5)
    self.bg_threshold = bg_threshold
    self._num_classes = num_classes

  def save_model(self, epoch):
      torch.save(self.state_dict(), "./fastRCNN_" + str(epoch) + ".pt")

  def load_model(self, path):
      self.load_state_dict(torch.load(path))

  # should it support batched images ?
  def forward(self, x):
    #if self.training is True:
    if isinstance(x, tuple):
      im, gt = x
    else:
      im = x
      gt = None
      id = None

    assert im.size(0) == 1, 'only single element batches supported'

    feats = self.features(_tovar(im))

    roi_boxes, rpn_prob, rpn_loss = self.rpn(im, feats, gt)
    if roi_boxes is None:
        if gt is not None:
            return None, None, None
        else:
            return None, None


    #if self.training is True:
    if gt is not None:
      # append gt boxes and sample fg / bg boxes
      # proposal_target-layer.py
      all_rois, frcnn_labels, roi_boxes, frcnn_bbox_targets, frcnn_viewpoints = self.frcnn_targets(roi_boxes, im, gt)

    # r-cnn
    regions = self.roi_pooling(feats, roi_boxes)
    scores, bbox_pred, viewpoint_pred = self.classifier(regions)

    boxes = self.bbox_reg(roi_boxes, bbox_pred, im)

    # apply cls + bbox reg loss here
    #if self.training is True:
    if gt is not None:
      frcnn_loss = self.frcnn_loss(scores, bbox_pred, viewpoint_pred, frcnn_labels, frcnn_bbox_targets, frcnn_viewpoints)
      loss = frcnn_loss + rpn_loss
      return loss, scores, boxes, viewpoint_pred

    return scores, boxes, viewpoint_pred

  def frcnn_loss(self, scores, bbox_pred, viewpoint_pred, labels, bbox_targets, viewpoints):
    cls_crit = nn.CrossEntropyLoss()
    #print(labels)
    if CUDA_AVAILABLE:
        cls_loss = cls_crit(scores, labels.long().cuda())
    else:
        cls_loss = cls_crit(scores, labels.long())

    # TODO: should be weight sum of diff. classes, not direct classification
    view_crit = nn.CrossEntropyLoss()
    if CUDA_AVAILABLE:
        view_loss = cls_crit(viewpoint_pred, labels.long().cuda())
    else:
        view_loss = cls_crit(viewpoint_pred, labels.long())


    reg_crit = nn.SmoothL1Loss()
    if CUDA_AVAILABLE:
        reg_loss = reg_crit(bbox_pred, bbox_targets.cuda())
    else:
        reg_loss = reg_crit(bbox_pred, bbox_targets)


    loss = cls_loss + reg_loss + view_loss
    print("Classfication loss is {:.3f}, Bbox loss is {:.3f}, and viewpoint pred loss is {:.3f}.".format(cls_loss.item(), reg_loss.item(), view_loss.item()))
    return loss

  def frcnn_targets(self, all_rois, im, gt):
    all_rois = all_rois.data.numpy()
    gt_boxes = gt['boxes'].numpy()
    gt_viewpoints = np.array(gt['viewpoints'])
    gt_labels = np.array(gt['gt_classes'])
    #zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    #all_rois = np.vstack(
    #    (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    #)
    all_rois = np.vstack((all_rois, gt_boxes))
    zeros = np.zeros((all_rois.shape[0], 1), dtype=all_rois.dtype)
    all_rois = np.hstack((zeros, all_rois))

    num_images = 1
    rois_per_image = self.batch_size / num_images
    fg_rois_per_image = np.round(self.fg_fraction * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, viewpoints = _sample_rois(self,
        all_rois, gt_boxes, gt_labels, gt_viewpoints, fg_rois_per_image,
        rois_per_image, self._num_classes)

    return _tovar((all_rois, labels, rois, bbox_targets, viewpoints))

  def bbox_reg(self, boxes, box_deltas, im):
    if CUDA_AVAILABLE:
        boxes = boxes.data[:,1:].cpu().numpy()
        box_deltas = box_deltas.data.cpu().numpy()
    else:
        boxes = boxes.data[:,1:].numpy()
        box_deltas = box_deltas.data.numpy()
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.size()[-2:])
    return _tovar(pred_boxes)

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        #print(ind, start, end)
        bbox_targets[ind, int(start):int(end)] = bbox_target_data[ind, 1:]
    return bbox_targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if False: #cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(self, all_rois, gt_boxes, gt_labels, gt_viewpoints, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    overlaps = overlaps.numpy()
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    #labels = gt_boxes[gt_assignment, 4]
    labels = gt_labels[gt_assignment]
    viewpoints = gt_viewpoints[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= self.fg_threshold)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < self.bg_threshold[1]) &
                       (max_overlaps >= self.bg_threshold[0]))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        #print(bg_inds, bg_rois_per_this_image)
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    viewpoints = viewpoints[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    # TODO: should background bbox's viewpoint be 0?
    #viewpoints[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, viewpoints
