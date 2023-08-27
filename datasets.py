import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import math

torch.manual_seed(17)
np.random.seed(0)

# ---------------------------------------------------------------------------------
# VIGOR

class VIGORDataset(Dataset):
    def __init__(self, root, label_root = 'splits_new', split='samearea', train=True, transform=None, pos_only=True, ori_noise=180):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only
        self.ori_noise = ori_noise
        
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
        print('Grd loaded, data size:{}'.format(self.data_size))
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
            grd = PIL.Image.new('RGB', (320, 640)) # if the image is unreadable, use a blank image
        grd = self.grdimage_transform(grd)
        
        # generate a random rotation 
        if self.ori_noise >= 180:
            rotation = np.random.uniform(low=0.0, high=1.0) # 
        else:
            rotation_range = self.ori_noise / 360
            rotation = np.random.uniform(low=-rotation_range, high=rotation_range)
        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation)*grd.size()[2]).int()).item(), dims=2)
                
        orientation_angle = rotation * 360 # 0 means heading North, counter-clockwise increasing
        
        # satellite
        if self.pos_only: # load positives only
            pos_index = 0
            sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
            [row_offset, col_offset] = self.delta[idx, pos_index] # delta = [delta_lat, delta_lon]
        else: # load positives and semi-positives
            col_offset = 320 
            row_offset = 320
            while (np.abs(col_offset)>=320 or np.abs(row_offset)>=320): # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0,3)
                sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
                [row_offset, col_offset] = self.delta[idx, pos_index] # delta = [delta_lat, delta_lon]

        sat = sat.convert('RGB')
        width_raw, height_raw = sat.size
        
        sat = self.satimage_transform(sat)
        _, height, width = sat.size()
        row_offset = np.round(row_offset/height_raw*height)
        col_offset = np.round(col_offset/width_raw*width)
        
        # groundtruth location on the aerial image       
        # Gaussian GT        
        gt = np.zeros([1, height, width], dtype=np.float32)
        gt_with_ori = np.zeros([20, height, width], dtype=np.float32)
        x, y = np.meshgrid(np.linspace(-width/2+col_offset,width/2+col_offset,width), np.linspace(-height/2-row_offset,height/2-row_offset,height))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        if self.train: 
            # find the ground truth orientation index, we use 20 orientation bins, and each bin is 18 degrees
            index = int(orientation_angle // 18)
            ratio = (orientation_angle % 18) / 18
            if index == 0:
                gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
                gt_with_ori[19, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
            else:
                gt_with_ori[20-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
                gt_with_ori[20-index-1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)
        
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
            
        return grd, sat, gt, gt_with_ori, orientation, city, orientation_angle

    
# ---------------------------------------------------------------------------------
# Oxford RobotCar

class OxfordRobotCarDataset(Dataset):
    def __init__(self, grd_image_root, 
                 sat_path, 
                 split='train', transform=None):
        self.grd_image_root = grd_image_root
        self.split = split
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]
            
        self.full_satellite_map = PIL.Image.open(sat_path) # meters_per_pixel: 0.09240351462361521
        
        # Load ground training or validation or test set
        self.grdList = []
        if self.split == 'train':
    
            with open(self.grd_image_root+'training.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    self.grdList.append(content.split(" "))

            with open(self.grd_image_root+'train_yaw.npy', 'rb') as f:
                self.grdYaw = np.load(f)
            
        elif self.split == 'val':
            with open(self.grd_image_root+'validation.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    self.grdList.append(content.split(" "))
            with open(self.grd_image_root+'val_yaw.npy', 'rb') as f:
                self.grdYaw = np.load(f)
      
        elif self.split == 'test':
            test_2015_08_14_14_54_57 = []
            with open(self.grd_image_root+'test1_j.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    test_2015_08_14_14_54_57.append(content.split(" "))
            test_2015_08_12_15_04_18 = []
            with open(self.grd_image_root+'test2_j.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    test_2015_08_12_15_04_18.append(content.split(" "))
            test_2015_02_10_11_58_05 = []
            with open(self.grd_image_root+'test3_j.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    test_2015_02_10_11_58_05.append(content.split(" "))
            
            self.test1_len = len(test_2015_08_14_14_54_57)
            self.test2_len = len(test_2015_08_12_15_04_18)
            self.test3_len = len(test_2015_02_10_11_58_05)
            
            self.grdList = test_2015_08_14_14_54_57 + test_2015_08_12_15_04_18 + test_2015_02_10_11_58_05
            
            with open(self.grd_image_root+'test_yaw.npy', 'rb') as f:
                self.grdYaw = np.load(f)

        self.grdNum = len(self.grdList)
        grdarray = np.array(self.grdList)
        self.grdUTM = np.transpose(grdarray[:,2:].astype(np.float64))
            
        # calculate the transformation from easting, northing to image col, row
        # transformation for the satellite image new
        primary = np.array([[619400., 5736195.],
                      [619400., 5734600.],
                      [620795., 5736195.],
                      [620795., 5734600.],
                      [620100., 5735400.]])
        secondary = np.array([[900., 900.], #tl
                    [492., 18168.], #bl
                    [15966., 1260.], #tr
                    [15553., 18528.], #br
                    [8255., 9688.]]) # c

        # Pad the data with ones, so that our transformation can do translations too
        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        X = pad(primary)
        Y = pad(secondary)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y)

        self.transform = lambda x: unpad(np.dot(pad(x), A))
    
    def __len__(self):
        return self.grdNum
    
    
    def __getitem__(self, idx):
        
        # ground
        grd = PIL.Image.open(os.path.join(self.grd_image_root, self.grdList[idx][0]))
        
        grd = grd.convert('RGB')
        grd = self.grdimage_transform(grd)
        
        image_coord = self.transform(np.array([[self.grdUTM[0, idx], self.grdUTM[1, idx]]]))[0] # pixel coords of the ground image. Easting, northing to image col, row
        
        if self.split == 'train':
            # generate a random offset for the ground image
            alpha = 2 * math.pi * random.random()
            r = 200 * np.sqrt(2) * random.random()
            row_offset = int(r * math.cos(alpha))
            col_offset = int(r * math.sin(alpha))

            sat_coord_row = int(image_coord[1] + row_offset)
            sat_coord_col = int(image_coord[0] + col_offset)
            
            sat = self.full_satellite_map.crop((sat_coord_col-400, sat_coord_row-400, sat_coord_col+400, sat_coord_row+400))
            
            
            row_offset_resized = int(np.round((400+row_offset)/800*512-256)) # ground location + offset = sat location
            col_offset_resized = int(np.round((400+col_offset)/800*512-256))
        
        if (self.split == 'val' or self.split == 'test'):
            col_split = int((image_coord[0]) // 400)
            if np.round(image_coord[0] - 400*col_split) <200:
                col_split -= 1
            col_pixel = int(np.round(image_coord[0] - 400*col_split))

            row_split = int((image_coord[1]) // 400)
            if np.round(image_coord[1] - 400*row_split) <200:
                row_split -= 1
            row_pixel = int(np.round(image_coord[1] - 400*row_split))

            sat = self.full_satellite_map.crop((col_split*400, row_split*400, col_split*400+800, row_split*400+800))
            
            
            row_offset_resized = int(-(row_pixel/800*512-256))
            col_offset_resized = int(-(col_pixel/800*512-256))
       
        sat = self.satimage_transform(sat)
        _, width, height = sat.size()
        
        # Gaussian GT
        x, y = np.meshgrid(np.linspace(-256+col_offset_resized,256+col_offset_resized,512), np.linspace(-256+row_offset_resized,256+row_offset_resized,512))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        ori = self.grdYaw[idx] # 0 means heading west, clockwise increasing, radian
        orientation_angle = (ori/np.pi*180) - 90  # 0 means heading north, clockwise increasing, degrees
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        
        gt_with_ori = np.zeros([20, height, width], dtype=np.float32)
        index = int(orientation_angle // 18)
        ratio = (orientation_angle % 18) / 18
        if index == 19:
            gt_with_ori[19, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        else:
            gt_with_ori[index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[index+1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)
        
        orientation = torch.full([2, height, width], np.cos(orientation_angle * np.pi/180))
        orientation[1,:,:] = np.sin(orientation_angle * np.pi/180)   
     
        return grd, sat, gt, gt_with_ori, orientation, orientation_angle

    
# ---------------------------------------------------------------------------------
# KITTI, our code is developed based on https://github.com/shiyujiao/HighlyAccurate
Default_lat = 49.015
Satmap_zoom = 18
SatMap_original_sidelength = 512 
SatMap_process_sidelength = 512 
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'  
left_color_camera_dir = 'image_02/data'
CameraGPS_shift_left = [1.08, 0.26]

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
        
        sat_rot = sat_map.rotate((-heading) / np.pi * 180) # make the east direction the vehicle heading
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR) 
        
        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        
        # randomly generate roation
        random_ori = np.random.uniform(-1, 1) * self.rotation_range # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
        
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
        

        # gt heat map
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x, y = np.meshgrid(np.linspace(-256+x_offset,256+x_offset,512), np.linspace(-256+y_offset,256+y_offset,512))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        # orientation gt
        orientation_angle = 90 - random_ori 
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
        
        gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        index = int(orientation_angle // 22.5)
        ratio = (orientation_angle % 22.5) / 22.5
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[15, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        else:
            gt_with_ori[16-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[16-index-1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)
        
        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi/180))
        orientation_map[1,:,:] = np.sin(orientation_angle * np.pi/180)
        
        
        return sat_map, grd_left_imgs[0], gt, gt_with_ori, orientation_map, orientation_angle
               
class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]
       

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
        
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR)
        
        # load the shifts 
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)

        random_ori = float(theta) * self.rotation_range # degree
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # gt heat map
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x, y = np.meshgrid(np.linspace(-256+x_offset,256+x_offset,512), np.linspace(-256+y_offset,256+y_offset,512))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        # orientation gt
        orientation_angle = 90 - random_ori 
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
            
        gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        index = int(orientation_angle // 22.5)
        ratio = (orientation_angle % 22.5) / 22.5
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[15, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        else:
            gt_with_ori[16-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[16-index-1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)
        
        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi/180))
        orientation_map[1,:,:] = np.sin(orientation_angle * np.pi/180)
        
        
        
        return sat_map, grd_left_imgs[0], gt, gt_with_ori, orientation_map, orientation_angle

    
    