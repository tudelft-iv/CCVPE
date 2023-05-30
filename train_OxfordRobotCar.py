import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "4" 

import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from datasets import OxfordRobotCarDataset
from losses import infoNCELoss, cross_entropy_loss, orientation_loss
from models import CVM_OxfordRobotCar as CVM
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

parser = argparse.ArgumentParser()
parser.add_argument('--training', choices=('True','False'), default='True')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--weight_ori', type=float, help='weight on orientation loss', default=1e1)
parser.add_argument('--weight_infoNCE', type=float, help='weight on infoNCE loss', default=1e4)

args = vars(parser.parse_args())
learning_rate = args['learning_rate']
batch_size = args['batch_size']
weight_ori = args['weight_ori']
weight_infoNCE = args['weight_infoNCE']
training = args['training'] == 'True'

label = 'OxfordRobotCar'


num_thread_workers = 1

grd_image_root='/scratch/zxia/datasets/Oxford_5m_sampling/'
sat_path='/scratch/zxia/datasets/Oxford_5m_sampling/satellite_map_new.png' 


transform_grd = transforms.Compose([
    transforms.Resize([154, 231]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

transform_sat = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])



if training == True:
    OxfordRobotCar_train = OxfordRobotCarDataset(grd_image_root, sat_path, split='train', transform=(transform_grd, transform_sat))
    OxfordRobotCar_val = OxfordRobotCarDataset(grd_image_root, sat_path, split='val', transform=(transform_grd, transform_sat))
    train_dataloader = DataLoader(OxfordRobotCar_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(OxfordRobotCar_val, batch_size=batch_size, shuffle=False)

else:
    OxfordRobotCar_test = OxfordRobotCarDataset(grd_image_root, sat_path, split='test', transform=(transform_grd, transform_sat))

    test1_indices = np.arange(0, OxfordRobotCar_test.test1_len)
    test1_set = Subset(OxfordRobotCar_test, test1_indices)
    test2_indices = np.arange(OxfordRobotCar_test.test1_len, OxfordRobotCar_test.test1_len+OxfordRobotCar_test.test2_len)
    test2_set = Subset(OxfordRobotCar_test, test2_indices)
    test3_indices = np.arange(OxfordRobotCar_test.test1_len+OxfordRobotCar_test.test2_len, OxfordRobotCar_test.grdNum)
    test3_set = Subset(OxfordRobotCar_test, test3_indices)
    
    test1_dataloader = DataLoader(test1_set, batch_size=batch_size, shuffle=False)
    test2_dataloader = DataLoader(test2_set, batch_size=batch_size, shuffle=False)
    test3_dataloader = DataLoader(test3_set, batch_size=batch_size, shuffle=False)

torch.cuda.empty_cache()
CVM_model = CVM(device)
if training:
    CVM_model.to(device)
    for param in CVM_model.parameters():
        param.requires_grad = True

    params = [p for p in CVM_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))

    global_step = 0
    # with torch.autograd.set_detect_anomaly(True):

    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        CVM_model.train()
        for i, data in enumerate(train_dataloader, 0):
            grd, sat, gt, gt_with_ori, gt_orientation, _ = [item.to(device) for item in data]

            gt_flattened = torch.flatten(gt, start_dim=1)
            gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

            gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori)
            gt_bottleneck2 = nn.MaxPool2d(32, stride=32)(gt_with_ori)
            gt_bottleneck3 = nn.MaxPool2d(16, stride=16)(gt_with_ori)
            gt_bottleneck4 = nn.MaxPool2d(8, stride=8)(gt_with_ori)
            gt_bottleneck5 = nn.MaxPool2d(4, stride=4)(gt_with_ori)
            gt_bottleneck6 = nn.MaxPool2d(2, stride=2)(gt_with_ori)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)

            loss_ori = torch.sum(torch.sum(torch.square(gt_orientation-ori), dim=1, keepdim=True) * gt) / logits_flattened.size()[0]        
            loss_infoNCE = infoNCELoss(torch.flatten(matching_score_stacked, start_dim=1), torch.flatten(gt_bottleneck, start_dim=1))
            loss_infoNCE2 = infoNCELoss(torch.flatten(matching_score_stacked2, start_dim=1), torch.flatten(gt_bottleneck2, start_dim=1))
            loss_infoNCE3 = infoNCELoss(torch.flatten(matching_score_stacked3, start_dim=1), torch.flatten(gt_bottleneck3, start_dim=1))
            loss_infoNCE4 = infoNCELoss(torch.flatten(matching_score_stacked4, start_dim=1), torch.flatten(gt_bottleneck4, start_dim=1))
            loss_infoNCE5 = infoNCELoss(torch.flatten(matching_score_stacked5, start_dim=1), torch.flatten(gt_bottleneck5, start_dim=1))
            loss_infoNCE6 = infoNCELoss(torch.flatten(matching_score_stacked6, start_dim=1), torch.flatten(gt_bottleneck6, start_dim=1))
            loss_ce =  cross_entropy_loss(logits_flattened, gt_flattened)
            loss = loss_ce + weight_infoNCE*(loss_infoNCE+loss_infoNCE2+loss_infoNCE3+loss_infoNCE4+loss_infoNCE5+loss_infoNCE6)/6 + weight_ori*loss_ori


            loss.backward()
            optimizer.step()

            global_step += 1
            # print statistics
            running_loss += loss.item()

            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        model_dir = 'models/OxfordRobotCar/'+label+'/' + str(epoch) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(CVM_model.cpu().state_dict(), model_dir+'model.pt') # saving model
        CVM_model.cuda() # moving model to GPU for further training
        CVM_model.eval()

        distance = []
        orientation_error = []
        for i, data in enumerate(val_dataloader, 0):
            grd, sat, gt, gt_with_ori, gt_orientation, _ = [item.to(device) for item in data]
            logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)
            gt = gt.cpu().detach().numpy() 
            gt_with_ori = gt_with_ori.cpu().detach().numpy() 
            gt_orientation = gt_orientation.cpu().detach().numpy() 
            heatmap = heatmap.cpu().detach().numpy()
            ori = ori.cpu().detach().numpy()
            for batch_idx in range(gt.shape[0]):
                current_gt = gt[batch_idx, :, :, :]
                loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                current_pred = heatmap[batch_idx, :, :, :]
                loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
                distance.append(np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)) 

                cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
                if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <=1:
                    a_acos_pred = math.acos(cos_pred)
                    if sin_pred < 0:
                        angle_pred = math.degrees(-a_acos_pred) % 360
                    else: 
                        angle_pred = math.degrees(a_acos_pred)
                    cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
                    a_acos_gt = math.acos(cos_gt)
                    if sin_gt < 0:
                        angle_gt = math.degrees(-a_acos_gt) % 360
                    else: 
                        angle_gt = math.degrees(a_acos_gt)
                    orientation_error.append(np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)]))      
    

        distance_error = np.mean(distance)
        print('epoch: ', epoch, 'mean distance error on validation set: ', distance_error)
        file = 'results/'+label+'_distance_error.txt'
        with open(file,'ab') as f:
            np.savetxt(f, [distance_error], fmt='%4f', header='validation_set_mean_distance_error_in_pixels:', comments=str(epoch)+'_')

        mean_orientation_error = np.mean(orientation_error)
        print('epoch: ', epoch, 'mean orientation error on validation set: ', mean_orientation_error)
        file = 'results/'+label+'_mean_orientation_error.txt'
        with open(file,'ab') as f:
            np.savetxt(f, [mean_orientation_error], fmt='%2f', header='validation_set_mean_orientation_error:', comments=str(epoch)+'_')
    

else:
    test_model_path = 'models/OxfordRobotCar/model.pt'
    test_model_path = '/scratch/zxia/checkpoints/TPAMI/main/TPAMI_Oxford_lr_0.0001_weight_ori_10.0/39/model.pt'
    
    print('load model from: ' + test_model_path)
    CVM_model.load_state_dict(torch.load(test_model_path))
    CVM_model.to(device)
    CVM_model.eval()

    meters_per_pixel = 0.09240351462361521
    distance_error = []
    longitudinal_error = []
    lateral_error = []
    orientation_error = []
    for i, data in enumerate(test1_dataloader, 0):
        grd, sat, gt, gt_with_ori, gt_orientation, orientation_angle = [item.to(device) for item in data]
        logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)
        gt = gt.cpu().detach().numpy() 
        gt_with_ori = gt_with_ori.cpu().detach().numpy() 
        gt_orientation = gt_orientation.cpu().detach().numpy() 
        orientation_angle = orientation_angle.cpu().detach().numpy() 
        heatmap = heatmap.cpu().detach().numpy()
        ori = ori.cpu().detach().numpy()
        for batch_idx in range(gt.shape[0]):
            current_gt = gt[batch_idx, :, :, :]
            loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
            current_pred = heatmap[batch_idx, :, :, :]
            loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
            gt_orientation_from_north = orientation_angle[batch_idx]
            pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
            distance_error.append(pixel_distance) 

            gt2pred_from_north = np.arctan2(np.abs(loc_gt[2]-loc_pred[2]), np.abs(loc_gt[1]-loc_pred[1])) * 180 / math.pi # degree
            angle_diff = np.abs(gt_orientation_from_north-gt2pred_from_north)
            longitudinal_error.append(np.abs(np.cos(angle_diff * np.pi/180) * pixel_distance))
            lateral_error.append(np.abs(np.sin(angle_diff * np.pi/180) * pixel_distance))


            cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
            if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <=1:
                a_acos_pred = math.acos(cos_pred)
                if sin_pred < 0:
                    angle_pred = math.degrees(-a_acos_pred) % 360
                else: 
                    angle_pred = math.degrees(a_acos_pred)
                cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
                a_acos_gt = math.acos(cos_gt)
                if sin_gt < 0:
                    angle_gt = math.degrees(-a_acos_gt) % 360
                else: 
                    angle_gt = math.degrees(a_acos_gt)
                orientation_error.append(np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)]))      

    print('test1')
    print('mean error (m): ', np.mean(distance_error) * meters_per_pixel / 512 * 800)
    print('median error (m): ', np.median(distance_error) * meters_per_pixel / 512 * 800)
    
    longitudinal_error_in_meters = np.array(longitudinal_error)  * meters_per_pixel / 512 * 800
    lateral_error_in_meters = np.array(lateral_error)  * meters_per_pixel / 512 * 800
    
    print('mean longitudinal error (m): ', np.mean(longitudinal_error_in_meters))
    print('median longitudinal error (m): ', np.median(longitudinal_error_in_meters))
    print('mean lateral error (m): ', np.mean(lateral_error_in_meters))
    print('median lateral error (m): ', np.median(lateral_error_in_meters))

    orientation_error = np.array(orientation_error)
    print('mean orientation error (deg): ', np.mean(orientation_error))
    print('median orientation error (deg): ', np.median(orientation_error))
    
    print('percentage of samples with longitudinal localization error under 1m, 3m, and 5m: ', np.sum(longitudinal_error_in_meters<1)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<3)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<5)/len(longitudinal_error_in_meters))
    print('percentage of samples with lateral localization error under 1m, 3m, and 5m: ', np.sum(lateral_error_in_meters<1)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<3)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<5)/len(lateral_error_in_meters))
    print('percentage of samples with orientation error under 1 degree, 3 degrees, and 5 degrees: ', np.sum(orientation_error<1)/len(orientation_error), np.sum(orientation_error<3)/len(orientation_error), np.sum(orientation_error<5)/len(orientation_error))

    print('----------------------------------------------------------------------')


    distance_error = []
    longitudinal_error = []
    lateral_error = []
    orientation_error = []
    for i, data in enumerate(test2_dataloader, 0):
        grd, sat, gt, gt_with_ori, gt_orientation, orientation_angle = [item.to(device) for item in data]
        logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)
        gt = gt.cpu().detach().numpy() 
        gt_with_ori = gt_with_ori.cpu().detach().numpy() 
        gt_orientation = gt_orientation.cpu().detach().numpy() 
        orientation_angle = orientation_angle.cpu().detach().numpy() 
        heatmap = heatmap.cpu().detach().numpy()
        ori = ori.cpu().detach().numpy()
        for batch_idx in range(gt.shape[0]):
            current_gt = gt[batch_idx, :, :, :]
            loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
            current_pred = heatmap[batch_idx, :, :, :]
            loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
            gt_orientation_from_north = orientation_angle[batch_idx]
            pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
            distance_error.append(pixel_distance) 

            gt2pred_from_north = np.arctan2(np.abs(loc_gt[2]-loc_pred[2]), np.abs(loc_gt[1]-loc_pred[1])) * 180 / math.pi # degree
            angle_diff = np.abs(gt_orientation_from_north-gt2pred_from_north)
            longitudinal_error.append(np.abs(np.cos(angle_diff * np.pi/180) * pixel_distance))
            lateral_error.append(np.abs(np.sin(angle_diff * np.pi/180) * pixel_distance))


            cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
            if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <=1:
                a_acos_pred = math.acos(cos_pred)
                if sin_pred < 0:
                    angle_pred = math.degrees(-a_acos_pred) % 360
                else: 
                    angle_pred = math.degrees(a_acos_pred)
                cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
                a_acos_gt = math.acos(cos_gt)
                if sin_gt < 0:
                    angle_gt = math.degrees(-a_acos_gt) % 360
                else: 
                    angle_gt = math.degrees(a_acos_gt)
                orientation_error.append(np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)]))      

    print('test2')
    print('mean error (m): ', np.mean(distance_error) * meters_per_pixel / 512 * 800)
    print('median error (m): ', np.median(distance_error) * meters_per_pixel / 512 * 800)
    
    longitudinal_error_in_meters = np.array(longitudinal_error)  * meters_per_pixel / 512 * 800
    lateral_error_in_meters = np.array(lateral_error)  * meters_per_pixel / 512 * 800
    
    print('mean longitudinal error (m): ', np.mean(longitudinal_error_in_meters))
    print('median longitudinal error (m): ', np.median(longitudinal_error_in_meters))
    print('mean lateral error (m): ', np.mean(lateral_error_in_meters))
    print('median lateral error (m): ', np.median(lateral_error_in_meters))

    orientation_error = np.array(orientation_error)
    print('mean orientation error (deg): ', np.mean(orientation_error))
    print('median orientation error (deg): ', np.median(orientation_error))

    print('percentage of samples with longitudinal localization error under 1m, 3m, and 5m: ', np.sum(longitudinal_error_in_meters<1)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<3)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<5)/len(longitudinal_error_in_meters))
    print('percentage of samples with lateral localization error under 1m, 3m, and 5m: ', np.sum(lateral_error_in_meters<1)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<3)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<5)/len(lateral_error_in_meters))
    print('percentage of samples with orientation error under 1 degree, 3 degrees, and 5 degrees: ', np.sum(orientation_error<1)/len(orientation_error), np.sum(orientation_error<3)/len(orientation_error), np.sum(orientation_error<5)/len(orientation_error))

    print('----------------------------------------------------------------------')

    distance_error = []
    longitudinal_error = []
    lateral_error = []
    orientation_error = []
    for i, data in enumerate(test3_dataloader, 0):
        grd, sat, gt, gt_with_ori, gt_orientation, orientation_angle = [item.to(device) for item in data]
        logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)
        gt = gt.cpu().detach().numpy() 
        gt_with_ori = gt_with_ori.cpu().detach().numpy() 
        gt_orientation = gt_orientation.cpu().detach().numpy() 
        orientation_angle = orientation_angle.cpu().detach().numpy() 
        heatmap = heatmap.cpu().detach().numpy()
        ori = ori.cpu().detach().numpy()
        for batch_idx in range(gt.shape[0]):
            current_gt = gt[batch_idx, :, :, :]
            loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
            current_pred = heatmap[batch_idx, :, :, :]
            loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
            gt_orientation_from_north = orientation_angle[batch_idx]
            pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
            distance_error.append(pixel_distance) 

            gt2pred_from_north = np.arctan2(np.abs(loc_gt[2]-loc_pred[2]), np.abs(loc_gt[1]-loc_pred[1])) * 180 / math.pi # degree
            angle_diff = np.abs(gt_orientation_from_north-gt2pred_from_north)
            longitudinal_error.append(np.abs(np.cos(angle_diff * np.pi/180) * pixel_distance))
            lateral_error.append(np.abs(np.sin(angle_diff * np.pi/180) * pixel_distance))


            cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
            if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <=1:
                a_acos_pred = math.acos(cos_pred)
                if sin_pred < 0:
                    angle_pred = math.degrees(-a_acos_pred) % 360
                else: 
                    angle_pred = math.degrees(a_acos_pred)
                cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
                a_acos_gt = math.acos(cos_gt)
                if sin_gt < 0:
                    angle_gt = math.degrees(-a_acos_gt) % 360
                else: 
                    angle_gt = math.degrees(a_acos_gt)
                orientation_error.append(np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)]))      

    print('test3')
    print('mean error (m): ', np.mean(distance_error) * meters_per_pixel / 512 * 800)
    print('median error (m): ', np.median(distance_error) * meters_per_pixel / 512 * 800)
    
    longitudinal_error_in_meters = np.array(longitudinal_error)  * meters_per_pixel / 512 * 800
    lateral_error_in_meters = np.array(lateral_error)  * meters_per_pixel / 512 * 800
    
    print('mean longitudinal error (m): ', np.mean(longitudinal_error_in_meters))
    print('median longitudinal error (m): ', np.median(longitudinal_error_in_meters))
    print('mean lateral error (m): ', np.mean(lateral_error_in_meters))
    print('median lateral error (m): ', np.median(lateral_error_in_meters))

    orientation_error = np.array(orientation_error)
    print('mean orientation error (deg): ', np.mean(orientation_error))
    print('median orientation error (deg): ', np.median(orientation_error))
    
    print('percentage of samples with longitudinal localization error under 1m, 3m, and 5m: ', np.sum(longitudinal_error_in_meters<1)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<3)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<5)/len(longitudinal_error_in_meters))
    print('percentage of samples with lateral localization error under 1m, 3m, and 5m: ', np.sum(lateral_error_in_meters<1)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<3)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<5)/len(lateral_error_in_meters))
    print('percentage of samples with orientation error under 1 degree, 3 degrees, and 5 degrees: ', np.sum(orientation_error<1)/len(orientation_error), np.sum(orientation_error<3)/len(orientation_error), np.sum(orientation_error<5)/len(orientation_error))


