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
import PIL.Image
from PIL import ImageFile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('--area', type=str, help='samearea or crossarea', default='samearea')
parser.add_argument('--pos_only', choices=('True','False'), default='True')
parser.add_argument('--ori_prior', type=float, help='prior in orientation, X means known orientation with +- X degree noise', default=180.)
parser.add_argument('--idx', type=int, help='image index')

args = vars(parser.parse_args())
area = args['area']
idx = args['idx']
ori_prior = args['ori_prior']
ori_prior = 18 * (ori_prior // 18) # round the closest multiple of 18 degrees within prior 
pos_only = args['pos_only'] == 'True'

dataset_root='/scratch/zxia/datasets/VIGOR'
test_model_path = 'models/VIGOR/samearea/model.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
np.random.seed(0)

class VIGORDataset(Dataset):
    def __init__(self, root, label_root = 'splits_new', split='samearea', train=False, transform=None, pos_only=True):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]
            
        if self.split == 'samearea':
            self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.split == 'crossarea':
            if self.train:
                self.city_list = ['NewYork', 'Seattle']
            else:
                self.city_list = ['SanFrancisco', 'Chicago']
        
        # load sat list
        self.sat_list = []
        self.sat_index_dict = {}

        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', sat_list_fname, idx)
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)
        print('Sat loaded, data size:{}'.format(self.sat_data_size))

        # load grd list  
        self.grd_list = []
        self.label = []
        self.sat_cover_dict = {}
        self.delta = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            if self.split == 'samearea':
                if self.train:
                    label_fname = os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt')
                else:
                    label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt')
            elif self.split == 'crossarea':
                label_fname = os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')
                
            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.label.append(label)
                    self.delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', label_fname, idx)
        self.data_size = len(self.grd_list)
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):        
        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert('RGB')
        except:
            print('unreadable image')
            grd = PIL.Image.new('RGB', (320, 640))    
            
        grd = self.grdimage_transform(grd)
        
        # generate a random rotation 
        rotation_range = ori_prior / 360
        rotation = np.random.uniform(low=-rotation_range, high=rotation_range)
        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation)*grd.size()[2]).int()).item(), dims=2)
                
        orientation_angle = rotation * 360 # 0 means heading North, clockwise increasing
        # satellite
        if self.pos_only:
            pos_index = 0
            sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
            [row_offset, col_offset] = self.delta[idx, pos_index] # delta = [delta_lat, delta_lon]
        else:
            col_offset = 320 # initialize it with a dummy number to enter the loop
            row_offset = 320
            while (np.abs(col_offset)>=320 or np.abs(row_offset)>=320):
                pos_index = random.randint(0,3)
                sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
                [row_offset, col_offset] = self.delta[idx, pos_index] # delta = [delta_lat, delta_lon]

        sat = sat.convert('RGB')
        width_raw, height_raw = sat.size
        
        sat = self.satimage_transform(sat)
        _, height, width = sat.size()
        row_offset = np.round(row_offset/height_raw*height)
        col_offset = np.round(col_offset/width_raw*width)
        
        # groundtruth location on the satellite map        
        # Gaussian GT        
        gt = np.zeros([1, height, width], dtype=np.float32)
        x, y = np.meshgrid(np.linspace(-width/2+col_offset,width/2+col_offset,width), np.linspace(-height/2-row_offset,height/2-row_offset,height))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        orientation = torch.full([2, height, width], np.cos(orientation_angle * np.pi/180))
        orientation[1,:,:] = np.sin(orientation_angle * np.pi/180)
        
        if 'NewYork' in self.grd_list[idx]:
            city = 'NewYork'
        elif 'Seattle' in self.grd_list[idx]:
            city = 'Seattle'
        elif 'SanFrancisco' in self.grd_list[idx]:
            city = 'SanFrancisco'
        elif 'Chicago' in self.grd_list[idx]:
            city = 'Chicago'
        
        return grd, sat, gt, orientation, city

    
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


vigor = VIGORDataset(dataset_root, split=area, train=False, pos_only=pos_only, transform=(transform_grd, transform_sat))    

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

torch.cuda.empty_cache()
CVM_model = CVM(device, ori_prior)
CVM_model.load_state_dict(torch.load(test_model_path))
CVM_model.to(device)
CVM_model.eval()

grd, sat, gt, orientation, city = vigor.__getitem__(idx)


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
plt.savefig('figures/'+area+'_'+str(idx)+'_pred_prior_'+str(ori_prior)+'.png', bbox_inches='tight', pad_inches=0)
print('Images are written to figures/')
    
    