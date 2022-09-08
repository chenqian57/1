#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import re

MAIN_DIR = "/mnt/disk1/lishenshen/" # default root for vocabulary files, model checkpoints, ranking files, heatmaps
# fahisoniq
# DATA_DIR = f'{MAIN_DIR}'
# VOCAB_DIR = f'{MAIN_DIR}/try2/vocab'
# CKPT_DIR = f'{MAIN_DIR}/try2/ckpt'
# RANKING_DIR = f'{MAIN_DIR}/try2/rankings'
# HEATMAP_DIR = f'{MAIN_DIR}/try2/heatmaps'

DATA_DIR = f'{MAIN_DIR}'
VOCAB_DIR = f'{MAIN_DIR}/tryraw_3810/vocab'
CKPT_DIR = f'{MAIN_DIR}/tryraw_3810/ckpt'
RANKING_DIR = f'{MAIN_DIR}/tryraw_3810/rankings'
HEATMAP_DIR = f'{MAIN_DIR}/tryraw_3810/heatmaps'
################################################################################
# *** Environment-related configuration
################################################################################

TORCH_HOME = "/mnt/disk1/lishenshen/pretrain_model/" # where ImageNet's pretrained models (resnet50/resnet18) weights are stored, locally on your machine
GLOVE_DIR = "/mnt/disk1/lishenshen/pretrain_model/" # where GloVe vectors (`glove.840B.300d.txt.pt`) are stored, locally on your machine

################################################################################
# *** Data paths
################################################################################

# FashionIQ
FASHIONIQ_IMAGE_DIR = f'{DATA_DIR}/images'
FASHIONIQ_ANNOTATION_DIR = f'{DATA_DIR}'

# Shoes
SHOES_IMAGE_DIR = f'{DATA_DIR}/shoes/images'
SHOES_ANNOTATION_DIR = f'{DATA_DIR}/shoes/annotations'

# CIRR
CIRR_IMAGE_DIR = f'{DATA_DIR}/cirr/img_feat_res152'
CIRR_ANNOTATION_DIR = f'{DATA_DIR}/cirr'

# Fashion200k
FASHION200K_IMAGE_DIR = f'{DATA_DIR}/women'
FASHION200K_ANNOTATION_DIR = f'{FASHION200K_IMAGE_DIR}/labels'

################################################################################
# *** OTHER
################################################################################

# Function to replace "/", "-" and "\" by a space and to remove all other caracters than letters or spaces (+ remove duplicate spaces)
cleanCaption = lambda cap : " ".join(re.sub('[^(a-zA-Z)\ ]', '', re.sub('[/\-\\\\]', ' ', cap)).split(" "))
