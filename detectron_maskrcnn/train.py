from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

from utils import *

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "object_detection/model_final.pth"

output_dir = "./object_detection"

num_classes = 1

device = "cuda"

train_dataset_name = "cloudphy_train"
train_images_path = "Step-1/images"

train_json_annot_path = "Step-1/labels.json"

test_dataset_name = "cloudphy_test"
test_images_path = "Step-1/images"

test_json_annot_path = "Step-1/labels.json"

cfg_save_path = "IS_cfg.pickle"

####################################
register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_images_path)

register_coco_instances(name=test_dataset_name, metadata={}, json_file=test_json_annot_path, image_root=test_images_path)

plot_samples(dataset_name=train_dataset_name, n=2)

# #########################333

def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
    from detectron2.data import build_detection_test_loader, DatasetMapper
    
    evaluator = COCOEvaluator(dataset_name="cloudphy_test", distributed=False, output_dir="Step-1")
    val_loader = build_detection_test_loader("cloudphy_test", mapper=DatasetMapper(is_train=False, augmentations=[], image_format="BGR"))
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == '__main__':
    main()
