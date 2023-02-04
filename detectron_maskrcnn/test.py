from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

import numpy as np
import pandas as pd

import cv2

cfg_save_path = "IS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1


dataset_path = "Step-1/images/"
data = pd.read_csv('Step-1/labels.csv')
data = pd.DataFrame.to_numpy(data)
# plot_samples("cloudphy_train")
# on_image(dataset_path + data[0][0], predictor)
predictor = DefaultPredictor(cfg)

for i in range(len(data)):
    img = cv2.imread(dataset_path + data[i][0])
    output = predictor(img)
    # print(output)
    # print(max((output['instances'].scores).tolist()))

    predictor = DefaultPredictor(cfg)
    on_image(dataset_path + data[i][0], predictor)
