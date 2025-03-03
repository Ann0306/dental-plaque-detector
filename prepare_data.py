import os
import shutil
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml



def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    데이터셋을 학습/검증/테스트 세트로 분할
    
    Args:
        data_dir (str): 데이터 디렉토리
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
    
    Returns:
        dict: 분할된 이미지 목록
    """
    # Check if ratios sum to 1.0 within a small tolerance to handle floating point precision issues
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        print(f"Warning: Ratios sum to {train_ratio + val_ratio + test_ratio}, adjusting to exactly 1.0")
        # Adjust test_ratio to make the sum exactly 1.0
        test_ratio = 1.0 - (train_ratio + val_ratio)
        
    image_dir = os.path.join(data_dir, 'images')
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 이미지 리스트 랜덤하게 섞기
    random.shuffle(all_images)
    
    # 분할 인덱스 계산
    n_images = len(all_images)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    # 데이터 분할
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]
    
    return {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

def organize_yolo_dataset(data_dir, output_dir, split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1}):
    """
    YOLO 형식으로 데이터셋 구성
    
    Args:
        data_dir (str): 원본 데이터 디렉토리
        output_dir (str): 출력 디렉토리
        split_ratios (dict): 데이터 분할 비율
    """
    # YOLO 구조 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # 데이터 분할
    data_splits = split_dataset(data_dir, split_ratios['train'], split_ratios['val'], split_ratios['test'])
    
    # 이미지 및 레이블 파일 복사
    for split, images in data_splits.items():
        print(f"Organizing {split} set...")
        for img_file in tqdm(images):
            # 이미지 파일 복사
            src_img = os.path.join(data_dir, 'images', img_file)
            dst_img = os.path.join(output_dir, split, 'images', img_file)
            shutil.copy(src_img, dst_img)
            
            # 레이블 파일 복사
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(data_dir, 'labels', label_file)
            dst_label = os.path.join(output_dir, split, 'labels', label_file)
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                # 레이블이 없는 경우 빈 파일 생성
                open(dst_label, 'w').close()
    
    # 데이터셋 YAML 파일 생성
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images'),
        'names': {
            0: 'tooth',
            1: 'plaque'
        }
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Dataset organized in YOLO format at: {output_dir}")
    print(f"Train: {len(data_splits['train'])} images")
    print(f"Validation: {len(data_splits['val'])} images")
    print(f"Test: {len(data_splits['test'])} images")

if __name__ == "__main__":
    # 예시 사용법
    
    # 1. 어노테이션을 YOLO 형식으로 변환 (필요한 경우)
    raw_data_dir = "raw_data"
    yolo_data_dir = "yolo_data"
    
    os.makedirs(os.path.join(yolo_data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_data_dir, 'labels'), exist_ok=True)
    
    class_mapping = {'tooth': 0, 'plaque': 1}
    
    if os.path.exists(raw_data_dir):
        print("Converting annotations to YOLO format...")
        
        image_files = [f for f in os.listdir(os.path.join(raw_data_dir, 'images')) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(raw_data_dir, 'images', img_file)
            
            # 어노테이션 파일 경로 (마스크 또는 JSON)
            mask_path = os.path.join(raw_data_dir, 'masks', base_name + '.png')
            
            if os.path.exists(mask_path):
                convert_annotation_to_yolo(img_path, mask_path, yolo_data_dir, class_mapping)
    
    # 2. 데이터셋 구성 및 분할
    output_dataset_dir = "data"
    
    # 이미 YOLO 형식으로 변환된 데이터가 있는 경우
    if os.path.exists(yolo_data_dir):
        print("Organizing dataset for YOLO training...")
        organize_yolo_dataset(
            data_dir=yolo_data_dir,
            output_dir=output_dataset_dir,
            # Ensure these values sum exactly to 1.0
            split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1}
        )
    
    print("Data preparation completed!") 