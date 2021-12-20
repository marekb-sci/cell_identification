# -*- coding: utf-8 -*-
import itertools
import io
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import sklearn

import torch
import torchvision
import torchmetrics


class DummyMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("value", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, value, weight=1.):

        self.value += value*weight
        self.weight += weight

    def compute(self):
        return self.value.float() / self.weight

class ImageLoader:
    """correctly load 16bit single channel image and tranform to torch.Tensor"""
    def __init__(self, depth=8):
        self.max_val = 2**depth -1

    def __call__(self, img_path):
        img_pil = Image.open(img_path)
        tensor = torch.Tensor(np.array(img_pil) / self.max_val)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

#%%
def log_confusion_matrix(cm, logger, class_names=None, num_classes=None,
                         image_label='confusion matrix', epoch=0):
    assert class_names is not None or num_classes is not None
    if class_names is None:
        class_names = list(map(str, range(num_classes)))
    num_classes = len(class_names)
    # cm = sklearn.metrics.confusion_matrix(gt_labels, pred_labels, labels= np.arange(num_classes), normalize='true')
    cm_figure = plot_confusion_matrix(cm, class_names)
    log_figure(logger, cm_figure, image_label=image_label, epoch=epoch)
    plt.close(cm_figure)

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes

    from: https://www.tensorflow.org/tensorboard/image_summaries
  """
  figure = plt.figure(figsize=(6, 6))
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
  labels = cm.astype(int)

  # Use white text if squares are dark; otherwise black.
  # threshold = cm.max() / 2.
  threshold = 0.5
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm_normalized[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


def log_figure(logger, fig, image_label='', epoch=0):
    """https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    pil_img = deepcopy(Image.open(buf))
    buf.close()

    img = torchvision.transforms.functional.to_tensor(pil_img)
    logger.add_image(image_label, img, epoch)

