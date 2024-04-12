#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is application show how to extract the accelerometer, gyroscope and magnetometer
measurements from  the Dataset of
"Towards Artefact Aware Human Motion Capture using Inertial Sensors Integrated into Loose Clothing"
https://zenodo.org/record/5948725


@author: michael lorenz, lorenz@cs.uni-kl.de
"""
import h5py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import os
from config import paths
from tqdm import tqdm
from articulate.model import ParametricModel

## configurations

# path to hte official SMPL model pickle file
SMPL_FILEPATH = '../data/body_models/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'     
# directory to the loose clothing dataset 
DIR_TO_H5 = '../data/clothing/'# -> Download the data From https://zenodo.org/record/5948725 <-
# 
SEQUENCES = ['shoulder_abduction', 'shoulder_flexion', 'squat', 'longterm']

# ABS_SEGMENTS = ['L-Forearm', 'L-Upperarm', 'L-Shoulder', 'R-Forearm', 'R-Upperarm', 'R-Shoulder', 'Sternum', 
#             'Pelvis', 'L-Tigh', 'L-Shank', 'R-Tigh', 'R-Shank', 'Head', 'L-Hand', 'R-Hand', 'L-Foot', 'R-Foot']

RE_ORDER = ['Pelvis', 'Sternum', 'Head', 
            'L-Tigh', 'L-Shank', 'L-Foot',
            'R-Tigh', 'R-Shank', 'R-Foot',
            'L-Shoulder', 'L-Upperarm', 'L-Forearm', 'L-Hand', 
            'R-Shoulder', 'R-Upperarm', 'R-Forearm', 'R-Hand']

CORRECTION = ['Pelvis', 'Sternum', 'Head', 
            'L-Shoulder', 'L-Upperarm', 'L-Forearm', 'L-Hand', 
            'R-Shoulder', 'R-Upperarm', 'R-Forearm', 'R-Hand']

PARENTS = [None, 0, 1, 0, 3, 4, 0, 6, 7, 1, 9, 10, 11, 1, 13, 14, 15]
# parents = [None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 
#            12, 13, 14, 16, 17, 18, 19, 20, 21]  # for smpl joints

imu_seg = ['L-Forearm','R-Forearm','L-Shank','R-Shank', 'Head', 'Pelvis']

smpl_joints = ['pelvis', 'l_hip', 'r_hip', 'spine_1', 'l_knee', 'r_knee', 'spine_2', 'l_ankle', 
               'r_ankle', 'spine_3', 'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 
               'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']

smpl_to_abs = {'pelvis': 'Pelvis', 
               'l_hip': 'L-Tigh', 
               'r_hip': 'R-Tigh', 
               'spine_1': 'Sternum', 
               'l_knee': 'L-Shank', 
               'r_knee': 'R-Shank', 
               'spine_2': None, 
               'l_ankle': 'L-Foot', 
               'r_ankle': 'R-Foot', 
               'spine_3': None, 
               'l_foot': None, 
               'r_foot': None, 
               'neck': 'Head', 
               'l_collar': 'L-Shoulder', 
               'r_collar': 'R-Shoulder', 
               'head': None, 
               'l_shoulder': 'L-Upperarm', 
               'r_shoulder': 'R-Upperarm', 
               'l_elbow': 'L-Forearm', 
               'r_elbow': 'R-Forearm', 
               'l_wrist': 'L-Hand', 
               'r_wrist': 'R-Hand', 
               'l_hand': None, 
               'r_hand': None}

dictPerson2Activ = {
# copied from https://github.com/lorenzcsunikl/Dataset-of-Artefact-Aware-Human-Motion-Capture-using-Inertial-Sensors-Integrated-into-Loose-Clothing/blob/main/utils/DataHelpers.py
 
    1: [# lifts pants very often
        (217,'NPose'), (500,'Walk'),
        (3491,'S1Shelf2Ground'), (3730,'Waiting'), (6600, 'S1Ground2Shelf'),
        (6900,'Walk'), (7653,'NPose'),(8106,'Walk'),
        (9398,'S2L-Hand-Up'),(10435,'S2R-Hand-Up'),(12157,'S2L-Hand-Low'),(13400,'S2R-Hand-Low'),
        (13846,'Walk'), (14546,'NPose'), (14708,'Walk'),
        (18287,'S3Working'),(18439,'NPose'),(21946,'S3Working'),
        (22204,'Walk'), (24403,'NPose'),(24660,'Walk'),
        (27760,'S1Shelf2Ground'), (27860,'Waiting'), (30702, 'S1Ground2Shelf'),
        (31092,'Walk'), (31627,'NPose'),(31925,'Walk'),
        (33415,'S2L-Hand-Up'),(34581,'S2R-Hand-Up'),(36138,'S2L-Hand-Low'),(37784,'S2R-Hand-Low'),
        (38284,'Walk'), (38668,'NPose'), (39000,'Walk'),
        (42705,'S3Working'),(42904,'NPose'),(46420,'S3Working'),
        (46622,'Walk'), (49706,'NPose'),(49970,'Walk'),
        (53016,'S1Shelf2Ground'), (55827,'Waiting'), (58990, 'S1Ground2Shelf'),
        (59300,'Walk'), (61930,'NPose'),(62226,'Walk'),
        (64375,'S2L-Hand-Up'),(65414,'S2R-Hand-Up'),(67077,'S2L-Hand-Low'),(68460,'S2R-Hand-Low'),
        (68822,'Walk'), (70024,'NPose'),(70140,'Walk'),
        (76240,'S3Working'),
        (76455,'Walk'), (79235,'NPose'),(79500,'Walk'),
        (82730,'S1Shelf2Ground'), (83020,'Waiting'), (86025, 'S1Ground2Shelf'),
        (86310,'Walk'), (86660,'NPose'),(86970,'Walk'),
        (88230,'S2L-Hand-Up'),(89300,'S2R-Hand-Up'),(90500,'S2L-Hand-Low'),(91500,'S2R-Hand-Low'),
        (91930,'Walk'), (92190,'NPose'),(92390,'Walk'),
        (99200,'S3Working'),
        (101036,'NPose')
        ],
    2: [ # lifts pants very often
        (320,'NPose'), (540,'Walk'),
        (3860,'S1Shelf2Ground'), (5270,'Waiting'), (8910, 'S1Ground2Shelf'),
        (9140,'Walk'), (10860,'NPose'),(11150,'Walk'),
        (12150,'S2L-Hand-Up'),(12700,'S2R-Hand-Up'),(13670,'S2L-Hand-Low'),(14770,'S2R-Hand-Low'),
        (15300,'Walk'),
        (22300,'S3Working'),
        (22570,'Walk'), (25280,'NPose'),(25440,'Walk'),
        (28180,'S1Shelf2Ground'), (29920,'Waiting'), (32880, 'S1Ground2Shelf'),
        (33120,'Walk'), (33780,'NPose'),(34110,'Walk'),
        (34890,'S2L-Hand-Up'),(35440,'S2R-Hand-Up'),(36540,'S2L-Hand-Low'),(37860,'S2R-Hand-Low'),
        (38200,'Walk'), (38790,'NPose'), (38930,'Walk'),
        (45150,'S3Working'),
        (45460,'Walk'), (47980,'NPose'),(48200,'Walk'),
        (51160,'S1Shelf2Ground'), (53890, 'S1Ground2Shelf'),
        (54330,'Walk'), (55730,'NPose'),(56020,'Walk'),
        (56730,'S2L-Hand-Up'),(57220,'S2R-Hand-Up'),(58160,'S2L-Hand-Low'),(59310,'S2R-Hand-Low'),
        (60670,'S2L-Hand-Up'),(61810,'S2R-Hand-Up'),(63310,'S2L-Hand-Low'),(65040,'S2R-Hand-Low'),
        (65460,'Walk'), (66800,'NPose'),(66920,'Walk'),
        (74230,'S3Working'),
        (74460,'Walk'), (77360,'NPose'),(77580,'Walk'),
        (80400,'S1Shelf2Ground'), (82800,'Waiting'), (85750, 'S1Ground2Shelf'),
        (86060,'Walk'), (87220,'NPose'),(87640,'Walk'),
        (88650,'S2L-Hand-Up'),(89500,'S2R-Hand-Up'),(90710,'S2L-Hand-Low'),(92120,'S2R-Hand-Low'),
        (93660,'NPose'),
        (94820,'S2L-Hand-Up'),(95750,'S2R-Hand-Up'),(97030,'S2L-Hand-Low'),(98370,'S2R-Hand-Low'),
        (98720,'Walk'), (99320,'NPose'),(99560,'Walk'),
        (106330,'S3Working'),(106620,'Walk'),(108053,'NPose'),
        ],
    3: [ # lifts pants very often
        (340,'NPose'), (690,'Walk'),
        (5680,'S1Shelf2Ground'), (8920, 'S1Ground2Shelf'),
        (9220,'Walk'), (11080,'NPose'),(11380,'Walk'),
        (12670,'S2L-Hand-Up'),(13760,'S2R-Hand-Up'),(14910,'S2L-Hand-Low'), (15980,'S2R-Hand-Low'),
        (16310,'Walk'), (16960,'NPose'), (17430,'Walk'),
        (24840,'S3Working'),
        (25040,'Walk'), (27540,'NPose'),(27750,'Walk'),
        (31200,'S1Shelf2Ground'), (34220,'Waiting'), (37490, 'S1Ground2Shelf'),
        (37760,'Walk'), (37980,'NPose'),(38480,'Walk'),
        (39180,'S2L-Hand-Up'),(39870,'S2R-Hand-Up'),(40740,'S2L-Hand-Low'),(42030,'S2R-Hand-Low'),
        (42270,'Walk'), (42920,'NPose'), (43160,'Walk'),
        (51160,'S3Working'),
        (51400,'Walk'), (54580,'NPose'),(54810,'Walk'),
        (57960,'S1Shelf2Ground'), (59760,'Waiting'), (63240, 'S1Ground2Shelf'),
        (63560,'Walk'), (65760,'NPose'),(66340,'Walk'),
        (67080,'S2L-Hand-Up'),(68040,'S2R-Hand-Up'),(69110,'S2L-Hand-Low'),(70350,'S2R-Hand-Low'),
        (70840,'Walk'), (72440,'NPose'),(72640,'Walk'),
        (80470,'S3Working'),
        (80690,'Walk'), (83020,'NPose'),(83380,'Walk'),
        (86680,'S1Shelf2Ground'), (89790, 'S1Ground2Shelf'),
        (90130,'Walk'), (91260,'NPose'), (92840,'Waiting'), (93200,'Walk'),
        (94080,'S2L-Hand-Up'),(95040,'S2R-Hand-Up'),(96190,'S2L-Hand-Low'),(97430,'S2R-Hand-Low'),
        (97700,'Walk'), (98520,'NPose'),(98565,'Walk'),
        (105870,'S3Working'),
        (106090,'Walk'),(106755,'NPose')
        ],
    4: [ # lifts pants often
        (130,'NPose'), (350,'Walk'),
        (3800,'S1Shelf2Ground'), (7470, 'S1Ground2Shelf'),
        (8560,'Walk'), (9320,'NPose'),(9600,'Walk'),
        (10850,'S2L-Hand-Up'),(11860,'S2R-Hand-Up'),(12890,'S2L-Hand-Low'),(14190,'S2R-Hand-Low'),
        (14460,'Walk'), (16520,'NPose'), (16730,'Walk'),
        (24390,'S3Working'),
        (24690,'Walk'), (25670,'NPose'),(25890,'Walk'),
        (29300,'S1Shelf2Ground'), (31920,'Waiting'), ( 35290, 'S1Ground2Shelf'),
        (35620,'Walk'), (36310,'NPose'),(36610,'Walk'),
        (37870,'S2L-Hand-Up'),(38810,'S2R-Hand-Up'),(39760,'S2L-Hand-Low'),(40880,'S2R-Hand-Low'),
        (41140,'Walk'), (41540,'NPose'), (41730,'Walk'),
        (49210,'S3Working'),
        (49480,'Walk'), (51710,'NPose'), (52010,'Walk'),
        (55410,'S1Shelf2Ground'), (58680, 'S1Ground2Shelf'),
        (58990,'Walk'), (60080,'NPose'),(60380,'Walk'),
        (61550,'S2L-Hand-Up'),(62550,'S2R-Hand-Up'),(63580,'S2L-Hand-Low'),(64470,'S2R-Hand-Low'),
        (64850,'Walk'), (65630,'NPose'),(65800,'Walk'),
        (73560,'S3Working'),
        (73870,'Walk'), (76720,'NPose'),(76960,'Walk'),
        (80340,'S1Shelf2Ground'), (83760, 'S1Ground2Shelf'),
        (84010,'Walk'), (85820,'NPose'),(86090,'Walk'),
        (87100,'S2L-Hand-Up'),(87890,'S2R-Hand-Up'),(88850,'S2L-Hand-Low'),(89700,'S2R-Hand-Low'),
        (90020,'Walk'), (92630,'NPose'),(92860,'Walk'),
        (100230,'S3Working'),
        (100460,'Walk'), (100566,'NPose')
        ],
    5 : [
        (30,'NPose'), (410,'Walk'),
        (4310,'S1Shelf2Ground'), (8030, 'S1Ground2Shelf'),
        (8630,'Walk'), (8880,'NPose'),(9120,'Walk'),
        (10400,'S2L-Hand-Up'),(11200,'S2R-Hand-Up'),(11950,'S2L-Hand-Low'),(12980,'S2R-Hand-Low'),
        (13390,'Walk'), (14940,'NPose'), (15150,'Walk'),
        (24640,'S3Working'),
        (24870,'Walk'), (26080,'Waiting'),(27610,'NPose'),(27810,'Waiting'),(28070,'Walk'),
        (32120,'S1Shelf2Ground'), (36820, 'S1Ground2Shelf'),
        (37120,'Walk'), (37840,'NPose'),(38040,'Walk'),
        (39090,'S2L-Hand-Up'),(40080,'S2R-Hand-Up'),(41010,'S2L-Hand-Low'),(42300,'S2R-Hand-Low'),
        (42740,'Walk'), (44420,'NPose'), (44650,'Walk'),
        (53840,'S3Working'),
        (54240,'Walk'), (57540,'NPose'),(57740,'Walk'),
        (61350,'S1Shelf2Ground'), (65870,'S1Ground2Shelf'),
        (66220,'Walk'), (68300,'NPose'),(68550,'Walk'),
        (69800,'S2L-Hand-Up'),(70760,'S2R-Hand-Up'),(71590,'S2L-Hand-Low'),(72670,'S2R-Hand-Low'),
        (73180,'Walk'), (74180,'NPose'),(74330,'Walk'),
        (83140,'S3Working'),
        (83620,'Walk'), (86620,'NPose'),(86950,'Walk'),
        (91010,'S1Shelf2Ground'), (92930,'Waiting'), (97190 ,'S1Ground2Shelf'),
        (97670,'Walk'), (98690,'NPose'),(99010,'Walk'),
        (100160,'S2L-Hand-Up'), (101180,'S2R-Hand-Up'), (102130,'S2L-Hand-Low'),(103200,'S2R-Hand-Low'),
        (103640,'Walk'), (103870,'NPose'),(104050,'Walk'),
        (113470,'S3Working'),
        (114000,'Walk'),(114128,'NPose')
        ],
    6: [
        (140,'NPose'), (430,'Walk'),
        (4640,'S1Shelf2Ground'), (6830,'Waiting'), (10890,'S1Ground2Shelf'),
        (11400,'Walk'), (12760,'NPose'),(13160,'Walk'),
        (15640,'S2L-Hand-Up'),(17160,'S2R-Hand-Up'),(19280,'S2L-Hand-Low'),(21440,'S2R-Hand-Low'),
        (21890,'Walk'), (22520,'NPose'), (22760,'Walk'),
        (30920,'S3Working'),
        (31220,'Walk'), (34390,'NPose'), (34690,'Walk'),
        (38540,'S1Shelf2Ground'), (42410, 'S1Ground2Shelf'),
        (43310,'Walk'), (44260,'NPose'),(44640,'Walk'),
        (46440,'S2L-Hand-Up'),(47600,'S2R-Hand-Up'),(48860,'S2L-Hand-Low'),(50460,'S2R-Hand-Low'),
        (50910,'Walk'), (52080,'NPose'), (52230,'Walk'),
        (61290,'S3Working'),
        (61540,'Walk'), (64460,'NPose'),(64720,'Walk'),
        (68810,'S1Shelf2Ground'), (72140,'Waiting'), (76010, 'S1Ground2Shelf'),
        (76220,'Walk'), (76660,'NPose'),(76990,'Walk'),
        (78300,'S2L-Hand-Up'),(79220,'S2R-Hand-Up'),(80560,'S2L-Hand-Low'),(81650,'S2R-Hand-Low'),
        (82040,'Walk'), (82370,'NPose'),(83420,'Walk'),
        (90790,'S3Working'),
        (91360,'Walk'), (92480,'NPose'),(92950,'Walk'),
        (95300,'S1Shelf2Ground'),( 97710, 'S1Ground2Shelf'),
        (98270,'Walk'), (98460,'NPose'),(98770,'Walk'),
        (99300,'S2L-Hand-Up'),(99810,'S2R-Hand-Up'),(100560,'S2L-Hand-Low'),(101320,'S2R-Hand-Low'),
        (101660,'Walk'), (101980,'NPose'),(102090,'Walk'),
        (108860,'S3Working'),
        (109150, 'Walk'), (109412,'NPose')
        ],
    7: [
        (40,'NPose'), (310,'Walk'),
        (3570,'S1Shelf2Ground'), (7410,'Waiting'), (10170, 'S1Ground2Shelf'),
        (10480,'Walk'), (11310,'NPose'),(11550,'Walk'),
        (13410,'S2L-Hand-Up'),(14610,'S2R-Hand-Up'),(16190,'S2L-Hand-Low'),(17520,'S2R-Hand-Low'),
        (17800,'Walk'), (18500,'NPose'), (18660,'Walk'),
        (24990,'S3Working'),
        (25320,'Walk'), (26440,'NPose'),(26760,'Walk'),
        (30050,'S1Shelf2Ground'), (32950,'Waiting'), (36020, 'S1Ground2Shelf'),
        (36470,'Walk'), (37830,'NPose'),(38190,'Walk'),
        (40350,'S2L-Hand-Up'),(41580,'S2R-Hand-Up'),(43620,'S2L-Hand-Low'),(45120,'S2R-Hand-Low'),
        (45500,'Walk'), ( 46180,'NPose'), (46330,'Walk'),
        (52620,'S3Working'),
        (53080,'Walk'), (54840,'NPose'),(55180,'Walk'),
        (58110,'S1Shelf2Ground'), (60600,'Waiting'), (63650, 'S1Ground2Shelf'),
        (64100,'Walk'), (65150,'NPose'),(65440,'Walk'),
        (67360,'S2L-Hand-Up'),(68520,'S2R-Hand-Up'),(70210,'S2L-Hand-Low'), (71440,'S2R-Hand-Low'),
        (71750,'Walk'), (75410,'NPose'),(75560,'Walk'),
        (82390,'S3Working'),
        (82660,'Walk'), (84640,'NPose'),(84850,'Walk'),
        (87910,'S1Shelf2Ground'), (90830,'Waiting'), ( 93560, 'S1Ground2Shelf'),
        (93980,'Walk'), (95520,'NPose'),(95730,'Walk'),
        (97570,'S2L-Hand-Up'),(98500,'S2R-Hand-Up'),(100100,'S2L-Hand-Low'),(101250,'S2R-Hand-Low'),
        (101600,'Walk'), (102960,'NPose'),(103130,'Walk'),
        (109260,'S3Working'),
        (109549,'NPose')
        ],
    8: [
        (120,'NPose'), (330,'Walk'),
        (3270,'S1Shelf2Ground'), (5990,'Waiting'), (8970, 'S1Ground2Shelf'),
        (9320,'Walk'), (10280,'NPose'),(10500,'Walk'),
        (12110,'S2L-Hand-Up'),(12960,'S2R-Hand-Up'),(14520,'S2L-Hand-Low'),(15860,'S2R-Hand-Low'),
        (16120,'Walk'), (17080,'NPose'), (17220,'Walk'),
        (22020,'S3Working'),
        (22240,'Walk'), (23660,'NPose'), (23870,'Walk'),
        (26340,'S1Shelf2Ground'), (28880, 'S1Ground2Shelf'),
        (29140,'Walk'), (29960,'NPose'),(30150,'Walk'),
        (31470,'S2L-Hand-Up'),(32300,'S2R-Hand-Up'),(33920,'S2L-Hand-Low'),(34970,'S2R-Hand-Low'),
        (35220,'Walk'), (37220,'NPose'), (37700,'Walk'),
        (42440,'S3Working'),
        (42580,'Walk'), (44120,'NPose'),(44370,'Walk'),
        (46880,'S1Shelf2Ground'), (50340,'Waiting'), (52930, 'S1Ground2Shelf'),
        (53250,'Walk'), (53740,'NPose'),(53900,'Walk'),
        (55270,'S2L-Hand-Up'),(56040,'S2R-Hand-Up'),(57630,'S2L-Hand-Low'),(58600,'S2R-Hand-Low'),
        (58940,'Walk'), (61730,'NPose'),(61860,'Walk'),
        (67500,'S3Working'),
        (67920,'Walk'), (69880,'NPose'),(70040,'Walk'),
        (72640,'S1Shelf2Ground'), (75400, 'S1Ground2Shelf'),
        (75720,'Walk'), (78180,'NPose'),(78370,'Walk'),
        (80090,'S2L-Hand-Up'), (80910,'S2R-Hand-Up'),(82410,'S2L-Hand-Low'),(83770,'S2R-Hand-Low'),
        (84290,'Walk'), (85370,'NPose'),(85510,'Walk'),
        (90980,'S3Working'),
        (91230,'Walk'),(92430,'NPose'),(92590,'Walk'),
        (95050,'S1Shelf2Ground'), (97510, 'S1Ground2Shelf'),
        (97730,'Walk'),(98180,'NPose'),(98390,'Walk'),
        (99660,'S2L-Hand-Up'), (100480,'S2R-Hand-Up'),(101990,'S2L-Hand-Low'),(103130,'S2R-Hand-Low'),
        (103490,'Walk'),(104110,'NPose'),(104120,'Walk'),
        (108660,'S3Working'),
        (108776,'NPose')
        ],
    9: [
        (130,'NPose'), (310,'Walk'),
        (3170,'S1Shelf2Ground'), (6130, 'S1Ground2Shelf'),
        (6390,'Walk'), (7790,'NPose'),(8020,'Walk'),
        (9680,'S2L-Hand-Up'),(11210,'S2R-Hand-Up'),(13080,'S2L-Hand-Low'),(14620,'S2R-Hand-Low'),
        (15010,'Walk'), (15830,'NPose'), (16000,'Walk'),
        (22250,'S3Working'),
        (22500,'Walk'), (24250,'NPose'),(24430,'Walk'),
        (27010,'S1Shelf2Ground'), (29710, 'S1Ground2Shelf'),
        (29950,'Walk'), (31880,'NPose'),(32140,'Walk'),
        (33930,'S2L-Hand-Up'),(35420,'S2R-Hand-Up'),(37110,'S2L-Hand-Low'),(38860,'S2R-Hand-Low'),
        (39230,'Walk'), (41630,'NPose'), (41770,'Walk'),
        (46270,'S3Working'),(47410,'Waiting'),(49390,'S3Working'),
        (49680,'Walk'), (51710,'NPose'),(51860,'Walk'),
        (55010,'S1Shelf2Ground'), (58550,'Waiting'), (61810, 'S1Ground2Shelf'),
        (62250,'Walk'), (66010,'NPose'),(66250,'Walk'),
        (67870,'S2L-Hand-Up'),(69540,'S2R-Hand-Up'),(71270,'S2L-Hand-Low'),(73310,'S2R-Hand-Low'),
        (73760,'Walk'), (77310,'NPose'),(77470,'Walk'),
        (84410,'S3Working'),
        (84720,'Walk'), (87590,'NPose'),(87780,'Walk'),
        (90980,'S1Shelf2Ground'), (95540,'Waiting'), (98730, 'S1Ground2Shelf'),
        (99050,'Walk'), (100900,'NPose'),(101160,'Walk'),
        (102990,'S2L-Hand-Up'),(104730,'S2R-Hand-Up'),(106270,'S2L-Hand-Low'),(108020,'S2R-Hand-Low'),
        (108390,'Walk'), (111870,'NPose'),(112050,'Walk'),
        (119110,'S3Working'),
        (119400,'Walk'), (119903,'NPose')
        ],
    10: [
        (100,'NPose'), (330,'Walk'),
        (3790,'S1Shelf2Ground'), (7060, 'S1Ground2Shelf'),
        (7360,'Walk'), (9380,'NPose'),(9640,'Walk'),
        (11410,'S2L-Hand-Up'),(12700,'S2R-Hand-Up'),(14610,'S2L-Hand-Low'),(16230,'S2R-Hand-Low'),
        (16680,'Walk'), (18080,'NPose'), (18240,'Walk'),
        (24360,'S3Working'),
        (24780,'Walk'), (26280,'NPose'),(26510,'Walk'),
        (29270,'S1Shelf2Ground'), (33370,'Waiting'), ( 35920, 'S1Ground2Shelf'),
        (36240,'Walk'), (38640,'Waiting'),(38950,'Walk'),
        (41090,'S2L-Hand-Up'),(42750,'S2R-Hand-Up'),(44710,'S2L-Hand-Low'),(46260,'S2R-Hand-Low'),
        (46700,'Walk'), (48380,'NPose'), (48530,'Walk'),
        (53570,'S3Working'),
        (53770,'Walk'), (56710,'NPose'),(56990,'Walk'),
        (60240,'S1Shelf2Ground'), (63290, 'S1Ground2Shelf'),
        (63570,'Walk'), (65120,'NPose'),(65340,'Walk'),
        (66720,'S2L-Hand-Up'),(67850,'S2R-Hand-Up'),(70220,'S2L-Hand-Low'),(71820,'S2R-Hand-Low'),
        (72180,'Walk'), (73660,'NPose'),(73810,'Walk'),
        (80060,'S3Working'),
        (80380,'Walk'), (84060,'NPose'),(84330,'Walk'),
        (87700,'S1Shelf2Ground'), (90520,'Waiting'), (93840, 'S1Ground2Shelf'),
        (94190,'Walk'), (94530,'NPose'),(97790,'Walk'),
        (99540,'S2L-Hand-Up'),(100720,'S2R-Hand-Up'),(102650,'S2L-Hand-Low'),(105470,'S2R-Hand-Low'),
        (105930,'Walk'), (106440,'NPose'),(106600,'Walk'),
        (113690,'S3Working'),
        (113890,'Walk'), (114053,'NPose')
        ]
}

def geodesic_angle(a: torch.Tensor, b: torch.Tensor):
    ''' as defined by Zhou et al in their paper
    '''
    # a = a.view(-1, 3, 3)
    # b = b.view(-1, 3, 3)
    c = torch.bmm(a.transpose(1,2), b)
    cos = (c.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) - 1.) / 2.
    cos = torch.clamp(cos, -1, 1)
    angles = torch.acos(cos)

    return angles 

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)

def vector_cross_matrix(x: torch.Tensor):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for each vector3 `v`. (torch, batch)

    :param x: Tensor that can reshape to [batch_size, 3].
    :return: The skew-symmetric matrix in shape [batch_size, 3, 3].
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)

def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r

def quaternion_to_angle_axis(quaternion):
    # Make sure the quaternion is normalized
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract the scalar part (w) and the vector part (x, y, z)
    w = quaternion[..., 0]
    vector_part = quaternion[..., 1:]

    # Compute the angle
    angles = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))

    # Prevent division by zero
    small_angle = angles < 1e-6
    sin_theta_half = torch.sqrt(1 - w * w)
    sin_theta_half[small_angle] = 1.0

    # Compute the axis
    axis = vector_part / sin_theta_half.unsqueeze(-1)

    # Handle the small angles separately
    axis[small_angle] = torch.zeros_like(vector_part[small_angle])

    # Return angle and axis as angle-axis (the magnitude of axis is the angle)
    angle_axis = axis * angles.unsqueeze(-1)
    
    return angle_axis

def quaternion_to_rotation_matrix(quaternion):
    # Ensure the quaternion is normalized
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract components of the quaternion
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Prepare components for the matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Compute rotation matrix components
    r11 = 1 - 2 * (yy + zz)
    r12 = 2 * (xy - wz)
    r13 = 2 * (xz + wy)
    r21 = 2 * (xy + wz)
    r22 = 1 - 2 * (xx + zz)
    r23 = 2 * (yz - wx)
    r31 = 2 * (xz - wy)
    r32 = 2 * (yz + wx)
    r33 = 1 - 2 * (xx + yy)

    # Stack components into matrix form
    rotation_matrix = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1).reshape(-1, 3, 3)
    
    return rotation_matrix.reshape(quaternion.shape[0], quaternion.shape[1], 3, 3)

def rotation_matrix_to_angle_axis(rotation_matrix):
    # Handle the case where the rotation is very close to 0
    # Set a small threshold
    epsilon = 1e-6

    # Calculate the angle
    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, min=-1.0, max=1.0))

    # Create a mask for the edge case where theta is very small (close to 0 or 2*pi)
    small_angle = theta.abs() < epsilon

    # Calculate each element of the axis vector
    u_x = rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]
    u_y = rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]
    u_z = rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]

    # Stack and normalize
    u = torch.stack([u_x, u_y, u_z], dim=-1) / (2 * torch.sin(theta).unsqueeze(-1))

    # Handle small angles separately
    u[small_angle] = torch.zeros_like(u[small_angle])

    # Multiply the axis by the angle to get the angle-axis representation
    angle_axis = u * theta.unsqueeze(-1)

    return angle_axis

def rotation_matrix_to_axis_angle_2(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result

def quaternion_to_axis_angle_2(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into axis-angles. (torch, batch)

    **Warning**: The returned axis angles may have a rotation larger than 180 degrees (in 180 ~ 360).

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    theta_half = q[:, 0].clamp(min=-1, max=1).acos()
    a = (q[:, 1:] / theta_half.sin().view(-1, 1) * 2 * theta_half.view(-1, 1)).view(-1, 3)
    a[torch.isnan(a)] = 0
    return a

def quaternion_to_rotation_matrix_2(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into rotation matrices. (torch, batch)

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                   2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                   2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
    return r.view(-1, 3, 3)

def q_inv(q):
    ''' assume quaternion of shape n x 4'''
    qi = q.clone()
    qi[:,1:] = -qi[:,1:]
    return qi

def quaternion_product(q1, q2):
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Calculate the product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack them into a new tensor
    return torch.stack((w, x, y, z), dim=-1)

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def inverse_kinematics_q(parents, global_quats):
    ''' inverse kinematics for quaternion representations
    '''
    local_quats = torch.zeros_like(global_quats)    # nj x n x 4
    for c, p in enumerate(parents):
        if c == 0:
            local_quats[0] = global_quats[0]
        else:
            local_quats[c] = qmul(q_inv(global_quats[p]), global_quats[c])
    return local_quats

def inverse_kinematics_r(parents, global_rots):
    local_rots = torch.zeros_like(global_rots)    # nj x n x 3 x 3
    for c, p in enumerate(parents):
        if c == 0:
            local_rots[0] = global_rots[0]
        else:
            local_rots[c] = torch.matmul(torch.linalg.inv(global_rots[p]), global_rots[c])
    return local_rots

def angle_between(a, b):
    offsets = a.transpose(1, 2).bmm(b)
    angles = rotation_matrix_to_angle_axis(offsets).norm(dim=1)
    return angles

def find_rotation_offset_torch(v1, v2):
    """
    Find the rotation matrix that aligns v2 to v1 in PyTorch.
    
    Parameters:
    - v1: A torch tensor of shape (3,), representing the target vector.
    - v2: A torch tensor of shape (3,), representing the source vector.
    
    Returns:
    - R: The rotation matrix that rotates v2 to align with v1.
    """
    # Ensure inputs are floats for accurate math operations
    v1 = v1.float()
    v2 = v2.float()
    
    # Normalize the input vectors
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)
    
    # Find the axis of rotation (cross product of v1 and v2)
    axis = torch.linalg.cross(v1, v2)
    axis_length = torch.norm(axis)
    if axis_length > 1e-6:  # To avoid division by zero
        axis = axis / axis_length
    
    # Find the angle of rotation (dot product of v1 and v2)
    angle = torch.arccos(torch.dot(v1, v2))
    
    # Using Rodrigues' rotation formula to compute the rotation matrix
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
    
    return R

def align_frame(acc_in, gyro_in, oris):
    ''' input: 
            acc and gyro as n x 3 torch tensors in local sensor frame
            orientation as n x 3 x 3 torch tensor in global bone frame
    '''
    # take the average acc reading while standing still
    n_window = 20
    a1 = acc_in[:n_window].mean(axis=0)
    r_global = oris[:n_window].transpose(1,2)   # inverse of R
    a2 = r_global.matmul(torch.tensor([0, -1., 0]))
    a2 = a2[:n_window].mean(axis=0)

    # find the offset
    r_offset = find_rotation_offset_torch(a1, a2)

    # apply on the data to get global 
    acc_bone = r_offset.matmul(acc_in.transpose(0,1)).transpose(0,1)
    acc_global = oris.matmul(acc_bone.unsqueeze(-1)).squeeze(-1)

    # TODO: remove gravity
    # acc_global -= torch.tensor([0, -9.8, 0])

    gyro_bone = r_offset.matmul(gyro_in.transpose(0,1)).transpose(0,1)
    gyro_global = oris.matmul(gyro_bone.unsqueeze(-1)).squeeze(-1)

    return acc_global, gyro_global

def _get_imu_data(h5_filename, test_subject):
    '''
    based on code from https://github.com/lorenzcsunikl/Dataset-of-Artefact-Aware-Human-Motion-Capture-using-Inertial-Sensors-Integrated-into-Loose-Clothing/blob/main/apps/mainGetIMUData.py
    '''
    h5 = h5py.File(h5_filename,'r')

    #get the order of the joints
    order_abs = [n.decode('utf-8')
                for n in list(h5['P'+str(test_subject)]['orderAbs'][...])]
    
    # select the subset of sensors 
    imu_idx = [order_abs.index(seg) for seg in imu_seg]

    # get the raw IMU reading
    accs = h5['P'+str(test_subject)]['acc'][...]   # 17 x n x 3
    gyrs = h5['P'+str(test_subject)]['gyr'][...]   # 17 x n x 3
    # mag = getValuesOf(h5, 'mag')[idx_segment_take,:,:]
    acc = torch.from_numpy(accs[imu_idx, :, :]).float().transpose(0,1)      # n x 6 x 3
    gyro = torch.from_numpy(gyrs[imu_idx, :, :]).float().transpose(0,1)
    quats_abs = h5['P'+str(test_subject)]['quatAbs'][...] # absolute orientation 

    h5.close()

    ori = quaternion_to_rotation_matrix(
        torch.from_numpy(quats_abs[imu_idx,:,:]).float())  # 6 x n x 3 x 3
    ori = ori.transpose(0,1)   # n x 6 x 3 x 3

    # rotation offset from the inertial frame to SMPL body frame
    r_it = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).float()  
    r1 = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0.]]).float()

    ori = r_it.matmul(ori)
    ori = r1.matmul(ori).matmul(r1.t())

    # fix the initial reading from the known bone angle
    # head
    # r_head = torch.linalg.inv(ori[0, 4])
    # ori[:,4] = ori[:,4].matmul(r_head)
    
    # left forearm
    r_left_elbow = R.from_euler('zx', [-80, -10], degrees=True)
    ori[:,0] = _rotate_bones(ori[:,0].clone(), r_left_elbow)    
    # right forearm
    r_right_elbow = R.from_euler('zx', [80, -10], degrees=True)
    ori[:,1] = _rotate_bones(ori[:,1].clone(), r_right_elbow)

    acc_global = acc.clone()
    gyro_global = gyro.clone()

    for i in range(6):
        acc_global[:,i], gyro_global[:,i] = align_frame(acc[:,i], gyro[:,i], ori[:,i])

    # return acc, gyro, ori
    return acc_global, gyro_global, ori


def _rotate_bones(joint_angles, r_init=R):
    '''
    provided the initial known bone angle, rotate the joint_angles to match the initial reading
    joint_angles: tensor of shape  n x 3 x 3
    r_init: scipy Rotation object, the initial rotation angle of the bone
    '''
    r_t = torch.tensor(r_init.as_matrix()).float()
    r_rot = torch.linalg.inv(joint_angles[0]).matmul(r_t)
    joint_angles = joint_angles.matmul(r_rot)
    return joint_angles

def npose_to_tpose_0(rots_global):
    '''
    Convert npose resting pose readings to tpose

    Parameters:
    - rots_global: tensor of shape 
        The rotation matrices of the joints in the npose resting pose.

    Returns:
    None
    '''
    # head
    r_head = torch.linalg.inv(rots_global[2, 0])
    rots_global[2] = rots_global[2].matmul(r_head)

    # right arm tree
    r_right_clavicle = R.from_euler('z', 10, degrees=True)
    rots_global[13] = _rotate_bones(rots_global[13].clone(), r_right_clavicle)
    r_right_shoulder = R.from_euler('z', 78, degrees=True)
    rots_global[14] = _rotate_bones(rots_global[14].clone(), r_right_shoulder)
    r_right_elbow = R.from_euler('zx', [90, -10], degrees=True)
    rots_global[15] = _rotate_bones(rots_global[15].clone(), r_right_elbow)
    r_right_hand = R.from_euler('zx', [90, -10], degrees=True)
    rots_global[16] = _rotate_bones(rots_global[16].clone(), r_right_hand)

    # left arm tree
    r_left_clavicle = R.from_euler('z', -10, degrees=True)
    rots_global[9] = _rotate_bones(rots_global[9].clone(), r_left_clavicle)
    r_left_shoulder = R.from_euler('z', -78, degrees=True)
    rots_global[10] = _rotate_bones(rots_global[10].clone(), r_left_shoulder)
    r_left_elbow = R.from_euler('zx', [-90, -10], degrees=True)
    rots_global[11] = _rotate_bones(rots_global[11].clone(), r_left_elbow)     
    r_left_hand = R.from_euler('zx', [-90, -10], degrees=True)
    rots_global[12] = _rotate_bones(rots_global[12].clone(), r_left_hand)
    
    return rots_global

def npose_to_tpose_1(rots_global):
    '''
    '''
    npose = torch.load('../data/npose.pt')
    body_model = ParametricModel(SMPL_FILEPATH, use_pose_blendshape=False)
    npose_global, _ = body_model.forward_kinematics(npose.view(1, 24, 3, 3))
    npose_global = npose_global[0]

    pos_global_mat = rots_global.clone()
    for i in range(24):
        seg_name = smpl_to_abs[smpl_joints[i]]
        if seg_name in CORRECTION:
            r_ref = R.from_matrix(npose_global[i].cpu().detach().numpy())
            j = RE_ORDER.index(seg_name)
            rots = rots_global[j]
            pos_global_mat[j] = _rotate_bones(rots.clone(), r_ref)     

    return pos_global_mat

def _get_smpl_pose(h5_filename, test_subject):

    h5 = h5py.File(h5_filename,'r')

    #get the order of the joints
    order_abs = [n.decode('utf-8')
                for n in list(h5['P'+str(test_subject)]['orderAbs'][...])]
    quats_abs = h5['P'+str(test_subject)]['quatAbs'][...] # absolute orientation \
    h5.close()
    
    # reorder
    order = [order_abs.index(seg) for seg in RE_ORDER]
    quats_global = torch.from_numpy(quats_abs[order,:,:]).float()
    rots_global = quaternion_to_rotation_matrix(quats_global)   # 17 x n x 3 x 3

    # rotation offset from the inertial frame to body frame
    r_it = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).float()  
    r1 = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0.]]).float()

    rots_global= r_it.matmul(rots_global)      # 17 x n x 3 x 3
    rots_global = r1.matmul(rots_global).matmul(r1.t())

    # make sure the root is aligned with the inertial frame
    r_root = rots_global[0,0]
    rots_global = torch.linalg.inv(r_root).matmul(rots_global)


    npose = torch.load('../data/npose.pt')
    smpl_file = '../data/body_models/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'     # official SMPL model path
    body_model = ParametricModel(smpl_file, use_pose_blendshape=False)
    npose_global, _ = body_model.forward_kinematics(npose.view(1, 24, 3, 3))
    npose_global = npose_global[0]

    pos_global_mat = rots_global.clone()
    for i in range(24):
        seg_name = smpl_to_abs[smpl_joints[i]]
        if seg_name in CORRECTION:
            r_ref = R.from_matrix(npose_global[i].cpu().detach().numpy())
            j = RE_ORDER.index(seg_name)
            rots = rots_global[j]
            pos_global_mat[j] = _rotate_bones(rots.clone(), r_ref)     

    imu_idx = [RE_ORDER.index(seg) for seg in imu_seg]
    length = rots_global.shape[1]

    pose = torch.zeros(24, length, 3)
    rots_local = inverse_kinematics_r(PARENTS, pos_global_mat)   # 17 x n x 3 x 3
    for i in range(24):
        seg_name = smpl_to_abs[smpl_joints[i]]
        if seg_name is not None:
            pose[i] = rotation_matrix_to_axis_angle_2(rots_local[RE_ORDER.index(seg_name)])
            
    # # convert to axis angle representation
    # pose = torch.zeros(24, length, 3)
    # pose_local_mat = torch.zeros(24, length, 3, 3)
    
    # npose = torch.load('../data/npose.pt')
    # for i in range(24):
    #     seg_name = smpl_to_abs[smpl_joints[i]]
    #     if seg_name is not None:
    #         pose_local_mat[i] = rots_local[RE_ORDER.index(seg_name)]
    #         pose[i] = rotation_matrix_to_axis_angle_2(pose_local_mat[i])
    #     else: 
    #         pose_local_mat[i] = torch.eye(3).repeat(length, 1, 1)
    

    pose = pose.transpose(0,1).reshape(-1, 72)   # n x 72

    return pose

def process_(split='test'):

    out_acc, out_ori, out_gyro, out_pose, out_tran = [], [], [], [], []
    out_acc_loose, out_ori_loose, out_gyro_loose = [], [], []
    out_pose_loose = []

    if split == 'train':
        test_subject = [1, 2, 3, 4, 5, 6, 7, 8]
    elif split == 'validation':
        test_subject = [9]
    elif split == 'test':
        test_subject = [10]
    elif split == 'all':
        test_subject = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f'---------- {split} split----------')
    for seq in tqdm(SEQUENCES):
        for ts in test_subject:
            # read the loose dataset
            filename_loose = DIR_TO_H5 + f'Loose_{seq}.h5'
            acc_loose, gyro_loose, ori_loose = _get_imu_data(filename_loose, ts)

            # read the tight dataset
            filename_tight = DIR_TO_H5 + f'Tight_{seq}.h5'
            acc, gyro, ori = _get_imu_data(filename_tight, ts)

            # get the gt pose
            pose = _get_smpl_pose(filename_tight, ts)

            if seq == 'longterm':
                time_activity = dictPerson2Activ[ts]
                beginning = 0
                for (end,act) in time_activity:
                    if act == 'NPose': 
                        continue    # merge the npose frames with the next one
                    out_acc.append(acc[beginning:end])
                    out_gyro.append(gyro[beginning:end])
                    out_ori.append(ori[beginning:end])
                    out_pose.append(pose[beginning:end])
                    out_pose_loose.append(pose[beginning:end])
                    out_tran.append(torch.zeros(acc.shape[0], 3))
                    out_acc_loose.append(acc_loose[beginning:end])
                    out_gyro_loose.append(gyro_loose[beginning:end])
                    out_ori_loose.append(ori_loose[beginning:end])
                    beginning = end
            else:
                out_acc.append(acc)
                out_gyro.append(gyro)
                out_pose.append(pose)
                out_pose_loose.append(pose)
                out_ori.append(ori)
                out_tran.append(torch.zeros(acc.shape[0], 3))
                out_acc_loose.append(acc_loose)
                out_gyro_loose.append(gyro_loose)
                out_ori_loose.append(ori_loose)

    # save to .pt files
    out_filename_tight = os.path.join(paths.tightimu_dir, f'{split}.pt')
    torch.save({'acc': out_acc, 'ori': out_ori, 
                'pose': out_pose, 'tran': out_tran, 
                'imu_sel': imu_seg, 'gyro': out_gyro,
                },
                out_filename_tight)
    print('Preprocessed data is saved at', out_filename_tight)

    out_filename_loose = os.path.join(paths.looseimu_dir, f'{split}.pt')
    torch.save({'acc': out_acc_loose, 'ori': out_ori_loose, 
                'pose': out_pose_loose, 'tran': out_tran, 
                'imu_sel': imu_seg, 'gyro': out_gyro_loose,
                },
                out_filename_loose)
    print('Preprocessed data is saved at', out_filename_loose)

process_(split='test')
process_(split='train')
process_(split='validation')
