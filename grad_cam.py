import os
import json

import torch
from PIL import Image
from torchvision import transforms,models
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import efficientnetv2

# coding: utf-8
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt



def draw_CAM(model, img_path, save_path, img_name):
    '''
    绘制 Class Activation Map
    :param model: Pytorch model
    :param img_path: Path
    :param save_path: Result Path
    :param transform: img preprocessing
    :param visual_heatmap: origin heatmap
    :return:
    '''
    img = Image.open(img_path).convert('RGB')
    img = img.resize((384, int(img.size[1] * 384 / img.size[0])))
    img = np.array(img, dtype=np.uint8)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = transform(img)
    input_tensor = torch.unsqueeze(img_tensor, 0)

    target_layers = [model.head[0]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img.astype(dtype=np.float32) / 255, grayscale_cam, use_rgb=False)
    print(img_name + " finished heatmap")
    cv2.imencode('.jpg', cam_image)[1].tofile(os.path.join(save_path, img_name))



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    img_path = "./dataset/tgt_gradcam/0"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img_path_list_jpg = [os.path.join(img_path, i) for i in os.listdir(img_path) if i.endswith(".jpg")]
    img_path_list_png = [os.path.join(img_path, i) for i in os.listdir(img_path) if i.endswith(".png")]
    img_path_list = img_path_list_jpg + img_path_list_png


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    num_classes = len(class_indict)

    # create model
    model = efficientnetv2.efficientnetv2_s(num_classes=num_classes).to(device)

    # load model weights
    model_weight_path = "./runs/eyePACS/efficientnetv2s.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    batch_size = 1
    for ids in range(0, len(img_path_list) // batch_size):
        for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
            assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
            draw_CAM(model, img_path, "../heatmap/tgt_domain/0",'gradcam ' + os.path.basename(img_path))
if __name__ == '__main__':
    main()
