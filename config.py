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
VOCAB_DIR = f'{MAIN_DIR}/tryraw_3810/vocab'   # fashion200K_vacab.pkl
CKPT_DIR = f'{MAIN_DIR}/tryraw_3810/ckpt'  #200K，消融实验，分层
RANKING_DIR = f'{MAIN_DIR}/tryraw_3810/rankings'  # 无
HEATMAP_DIR = f'{MAIN_DIR}/tryraw_3810/heatmaps'  # 无
################################################################################
# *** Environment-related configuration 环境相关配置
################################################################################

TORCH_HOME = "/mnt/disk1/lishenshen/pretrain_model/" # where ImageNet's pretrained models (resnet50/resnet18) weights are stored, locally on your machine
# ImageNet的预训练模型(resnet50/resnet18)权重存储，本地在机器上
GLOVE_DIR = "/mnt/disk1/lishenshen/pretrain_model/" # where GloVe vectors (`glove.840B.300d.txt.pt`) are stored, locally on your machine
# GloVe vectors (`glove.840B.300d.txt.pt`)存储的位置，本地在您的机器上
# glove.840B.300d.txt.pt是GloVe算法生成的词向量文件之一，包含840B个单词的300维向量表示。
# 也可以使用其他的词向量文件，例如glove.6B.300d.txt
################################################################################
# *** Data paths
################################################################################

# FashionIQ
FASHIONIQ_IMAGE_DIR = f'{DATA_DIR}/IQ/images'  # /mnt/disk1/lishenshen/IQ/images    # f'{DATA_DIR}/images' 
FASHIONIQ_ANNOTATION_DIR = f'{DATA_DIR}/IQ/captions'    # /mnt/disk1/lishenshen/IQ/captions   # f'{DATA_DIR}' 

# Shoes
SHOES_IMAGE_DIR = f'{DATA_DIR}/shoes/images'  # 
SHOES_ANNOTATION_DIR = f'{DATA_DIR}/shoes/annotations'  # 

# CIRR
CIRR_IMAGE_DIR = f'{DATA_DIR}/img_feat_res152'  # /mnt/disk1/lishenshen/img_feat_res152   # f'{DATA_DIR}/cirr/img_feat_res152'
CIRR_ANNOTATION_DIR = f'{DATA_DIR}/cirr'

# Fashion200k
FASHION200K_IMAGE_DIR = f'{DATA_DIR}/women'   
FASHION200K_ANNOTATION_DIR = f'{FASHION200K_IMAGE_DIR}/labels'

################################################################################
# *** OTHER
################################################################################

# Function to replace "/", "-" and "\" by a space and to remove all other caracters than letters or spaces (+ remove duplicate spaces)
# 函数将“/”、“-”和“\”替换为空格，并删除除字母或空格以外的所有其他字符，（删除重复空格）
cleanCaption = lambda cap : " ".join(re.sub('[^(a-zA-Z)\ ]', '', re.sub('[/\-\\\\]', ' ', cap)).split(" "))
