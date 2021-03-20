import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models.hrnet import HRNet
from models.poseresnet import PoseResNet
from models.stackedhourglass import hg
from models.predictor import HumanPosePredictor

class SimpleHPNet:
    """
    SimpleHPNet class

    provides an intergration of HRnet, Simple Baselines hrnet
    and Stacked Hourglass Net
    """

    def __init__(self, 
                 c, 
                 nof_joints, 
                 checkpoint_path, 
                 model_name = 'HRNet', 
                 resolution = (384, 288), 
                 device=torch.device('cuda')
                 ):
        
        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        elif model_name in ('hg', 'HG'):
            self.model = hg(num_stacks=c, num_blocks=1, num_classes=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image):
        if len(image.shape) == 3:
            return self._predict_single(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        if self.model_name in ('HRNet', 'hrnet', 'PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            pts = self._predict_single_hrnet_resnet(image)
        else:
            pts = self._predict_single_hg(image)

        return pts

    def _predict_single_hg(self, image):
        self.model = HumanPosePredictor(self.model)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
        
        pts = self.model.estimate_joints(image, flip=False)
        return pts


    def _predict_single_hrnet_resnet(self, image):
        old_res = image.shape
        if self.resolution is not None:
            image = cv2.resize(
                image,
                (self.resolution[1], self.resolution[0])
            )

        images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
        boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]
        heatmaps = np.zeros((1, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        images = images.to(self.device)
        with torch.no_grad():
            out = self.model(images)
        out = out.detach().cpu().numpy()

        pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
        # For each human, for each joint: y, x, confidence
        for i, human in enumerate(out):
            heatmaps[i] = human
            for j, joint in enumerate(human):
                pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                # 2: confidences
                pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                pts[i, j, 2] = joint[pt]
        return pts

   



