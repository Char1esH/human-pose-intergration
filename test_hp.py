import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from SimpleHPNet import SimpleHPNet
from models.utils.visualization import joints_dict, draw_points_and_skeleton

#model = SimpleHPNet(48, 17, "weights/hrnet/pose_hrnet_w48_384x288.pth", model_name='hrnet')
#model = SimpleHPNet(101, 17, "weights/resnet/pose_resnet_101_384x288.pth", model_name='poseresnet')
model = SimpleHPNet(8, 16, 'weights/hg/bearpaw_hg8.pth', model_name='hg', resolution=(256, 256))


image = cv2.imread("./test_img/test_img002.jpg", cv2.IMREAD_COLOR)
print(image.shape)

pts = model.predict(image)
pts = pts[0]
print(pts)
print(pts.shape)

new_image = draw_points_and_skeleton(image, pts, joints_dict()['mpii']['skeleton'], 
                                     person_index = 0, confidence_threshold = 0.3, 
                                     points_color_palette='gist_rainbow', skeleton_color_palette='jet', 
                                     points_palette_samples=8)

# new_image = draw_points_and_skeleton(image, pts, joints_dict()['coco']['skeleton'], 
#                                      person_index = 0, confidence_threshold = 0.3, 
#                                      points_color_palette='gist_rainbow', skeleton_color_palette='jet', 
#                                      points_palette_samples=8)

cv2.imwrite('result.jpg', new_image)