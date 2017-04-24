from matplotlib import pyplot
import numpy as np
import os
import shutil
from IPython import display

from caffe2.python import core, cnn, net_drawer, workspace, visualize

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
caffe2_root = '~/caffe2'


