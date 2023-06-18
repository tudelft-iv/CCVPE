import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# os.environ["MKL_NUM_THREADS"] = "4" 
# os.environ["NUMEXPR_NUM_THREADS"] = "4" 
# os.environ["OMP_NUM_THREADS"] = "4" 

import argparse
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from models import CVM_VIGOR_ori_prior as CVM
from datasets import VIGORDataset
import PIL.Image
from PIL import ImageFile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('--area', type=str, help='samearea or crossarea', default='samearea')
parser.add_argument('--pos_only', choices=('True','False'), default='True')
parser.add_argument('--ori_noise', type=float, help='noise in orientation prior, 180 means unknown orientation', default=180.)
parser.add_argument('--idx', type=int, help='image index')

args = vars(parser.parse_args())
area = args['area']
idx = args['idx']
ori_noise = args['ori_noise']
ori_noise = 18 * (ori_noise // 18) # round the closest multiple of 18 degrees within prior 
pos_only = args['pos_only'] == 'True'

dataset_root='/scratch/zxia/datasets/VIGOR'
test_model_path = 'models/VIGOR/samearea/model.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
np.random.seed(0)

    
transform_grd = transforms.Compose([
    transforms.Resize([320, 640]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

transform_sat = transforms.Compose([
    # resize
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])


vigor = VIGORDataset(dataset_root, split=area, train=False, pos_only=pos_only, transform=(transform_grd, transform_sat), ori_noise=ori_noise)

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

torch.cuda.empty_cache()
CVM_model = CVM(device, ori_noise)
CVM_model.load_state_dict(torch.load(test_model_path))
CVM_model.to(device)
CVM_model.eval()

grd, sat, gt, _, orientation, city, _ = vigor.__getitem__(idx)


grd_feed = grd.unsqueeze(0)
sat_feed = sat.unsqueeze(0)

grd_feed = grd_feed.to(device)
sat_feed = sat_feed.to(device)
grd = invTrans(grd)
sat = invTrans(sat)

logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd_feed, sat_feed)
matching_score_max1, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
matching_score_max2, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
matching_score_max3, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
matching_score_max4, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
matching_score_max5, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
matching_score_max6, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)

# grd = grd.cpu().detach().numpy() 
# sat = sat.cpu().detach().numpy() 
gt = gt.permute(1, 2, 0)
gt = gt.cpu().detach().numpy() 
loc_gt = np.unravel_index(gt.argmax(), gt.shape)


orientation = orientation.permute(1, 2, 0).cpu().detach().numpy() 

heatmap = torch.squeeze(heatmap, dim=0).permute(1, 2, 0)
heatmap = heatmap.cpu().detach().numpy()
loc_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)
ori = torch.squeeze(ori, dim=0).permute(1, 2, 0)
ori = ori.cpu().detach().numpy()


cos_pred_dense = ori[:, :, 0]
sin_pred_dense = ori[:, :, 1]
cos_pred, sin_pred = ori[loc_pred[0], loc_pred[1], :]


cos_gt, sin_gt = orientation[loc_gt[0], loc_gt[1], :]
a_acos_gt = math.acos(cos_gt)
if sin_gt < 0:
    angle_gt = math.degrees(-a_acos_gt) % 360
else: 
    angle_gt = math.degrees(a_acos_gt)

plt.figure(figsize=(8,12))
plt.imshow(  grd.permute(1, 2, 0)  )
plt.axvline(grd.size()[2]/2, color='g')
plt.axis('off')
plt.savefig('figures/'+area+'_'+str(idx)+'_grd_'+'.png', bbox_inches='tight', pad_inches=0)

#     plt.figure(figsize=(16,10))
#     plt.subplot(2,3,1)
#     plt.imshow(torch.squeeze(matching_score_max1, dim=0).permute(1, 2, 0).cpu().detach().numpy()  )
#     plt.subplot(2,3,2)
#     plt.imshow(torch.squeeze(matching_score_max2, dim=0).permute(1, 2, 0).cpu().detach().numpy()  )
#     plt.subplot(2,3,3)
#     plt.imshow(torch.squeeze(matching_score_max3, dim=0).permute(1, 2, 0).cpu().detach().numpy()  )
#     plt.subplot(2,3,4)
#     plt.imshow(torch.squeeze(matching_score_max4, dim=0).permute(1, 2, 0).cpu().detach().numpy()  )
#     plt.subplot(2,3,5)
#     plt.imshow(torch.squeeze(matching_score_max5, dim=0).permute(1, 2, 0).cpu().detach().numpy()  )
#     plt.subplot(2,3,6)
#     plt.imshow(torch.squeeze(matching_score_max6, dim=0).permute(1, 2, 0).cpu().detach().numpy()  )

plt.figure(figsize=(6,6))
plt.imshow(  sat.permute(1, 2, 0)  )
plt.imshow(heatmap,  norm=LogNorm(vmax=np.max(heatmap)), alpha=0.6, cmap='Reds')
plt.scatter(loc_gt[1], loc_gt[0], s=300, marker='^', facecolor='g', label='GT', edgecolors='white')
plt.scatter(loc_pred[1], loc_pred[0], s=300, marker='*', facecolor='gold', label='Ours', edgecolors='white')
xx,yy = np.meshgrid(np.linspace(0,512,512),np.linspace(0,512,512))
cos_angle = ori[:,:,0]
sin_angle = ori[:,:,1]
plt.quiver(xx[::40, ::40], yy[::40, ::40], -sin_pred_dense[::40, ::40], cos_pred_dense[::40, ::40], linewidths=0.2, scale=14, width=0.01) # plot the predicted rotation angle + 90 degrees
plt.quiver(loc_pred[1], loc_pred[0], -sin_pred, cos_pred, color='gold', linewidths=0.2, scale=10, width=0.015)
plt.quiver(loc_gt[1], loc_gt[0], -np.sin(angle_gt / 180 * np.pi), np.cos(angle_gt / 180 * np.pi), color='g', linewidths=0.2, scale=10, width=0.015)
plt.axis('off')
plt.legend(loc=2, framealpha=0.8, labelcolor='black', prop={'size': 15})
plt.savefig('figures/'+area+'_'+str(idx)+'_noise_in_orientation_'+str(ori_noise)+'.png', bbox_inches='tight', pad_inches=0)
print('Images are written to figures/')
    
    