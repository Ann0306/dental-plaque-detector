import os
import yaml
from ultralytics import YOLO
import torch
from datetime import datetime
import shutil
from pathlib import Path

def train_model(data_yaml_path, epochs=100, img_size=640, batch_size=16, workers=4):
    """
    치아 치석 감지를 위한 YOLOv8 segmentation 모델 fine-tuning
    
    Args:
        data_yaml_path (str): 데이터 설정 YAML 파일 경로
        epochs (int): 학습 에폭 수
        img_size (int): 입력 이미지 크기
        batch_size (int): 배치 크기
        workers (int): 데이터 로딩을 위한 워커 수
    
    Returns:
        tuple: (model, results) - 학습된 모델과 학습 결과
    """
    # GPU 사용 가능 여부 확인
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 실험 시간 기록
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'dental_plaque_{timestamp}'
    
    try:
        # YOLOv8-seg 모델 로드 (nano 버전)
        model = YOLO('yolov8n-seg.pt')
        print("Successfully loaded pretrained YOLOv8-seg model")
        
        # 모델 설정
        model_config = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'workers': workers,
            'patience': 30,            # Early stopping patience
            'save': True,              # 모델 저장
            'project': 'models',       # 저장 디렉토리
            'name': experiment_name,   # 실험 이름
            'exist_ok': True,          # 기존 실험 덮어쓰기
            'pretrained': True,        # 사전 학습 가중치 사용
            'optimizer': 'AdamW',      # 개선된 옵티마이저
            'lr0': 0.001,             # 초기 학습률
            'lrf': 0.01,              # 최종 학습률 (cosine scheduler)
            'momentum': 0.937,         # SGD 모멘텀
            'weight_decay': 0.0005,    # 가중치 감쇠
            'warmup_epochs': 3.0,      # 준비 운동 에폭
            'warmup_momentum': 0.8,    # 준비 운동 모멘텀
            'warmup_bias_lr': 0.1,     # 준비 운동 편향 학습률
            'box': 7.5,                # 박스 손실 가중치
            'cls': 0.5,                # 클래스 손실 가중치
            'dfl': 1.5,                # DFL 손실 가중치
            'device': device,          # 학습 장치
            'plots': True,             # 결과 시각화
            'save_period': 10,         # 몇 에폭마다 저장할지
            'amp': True,               # 자동 혼합 정밀도
            'mask_ratio': 4,           # 마스크 해상도 비율
            'overlap_mask': True,      # 마스크 오버랩 허용
            'val': True,               # 검증 수행
        }
        
        # 모델 학습
        results = model.train(**model_config)
        
        # 최종 모델 복사
        best_model_path = os.path.join('models', experiment_name, 'weights', 'best.pt')
        final_model_path = os.path.join('models', 'best.pt')
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, final_model_path)
            print(f"Best model saved to {final_model_path}")
        
        return model, results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

def create_data_yaml(train_path, val_path, test_path=None, num_classes=2, class_names=None):
    """
    YOLO 학습을 위한 데이터 YAML 파일 생성
    
    Args:
        train_path (str): 학습 데이터 경로
        val_path (str): 검증 데이터 경로
        test_path (str, optional): 테스트 데이터 경로
        num_classes (int): 클래스 수
        class_names (list): 클래스 이름 목록
    
    Returns:
        str: YAML 파일 경로
    """
    if class_names is None:
        class_names = ['tooth', 'plaque']
    
    # 절대 경로 사용
    data = {
        'path': os.path.abspath('./data'),  # 데이터 루트 디렉토리
        'train': os.path.join('train', 'images'),  # 학습 이미지 경로
        'val': os.path.join('val', 'images'),      # 검증 이미지 경로
        'test': os.path.join('test', 'images'),    # 테스트 이미지 경로
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': num_classes  # 클래스 수 명시
    }
    
    os.makedirs('models', exist_ok=True)
    yaml_path = os.path.join('models', 'dental_plaque.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Data YAML created at {yaml_path}")
    return yaml_path

def validate_dataset_structure():
    """
    데이터셋 구조 검증 및 파일 수 확인
    """
    required_dirs = [
        'data/train/images',
        'data/train/labels',
        'data/val/images',
        'data/val/labels',
        'data/test/images',
        'data/test/labels'
    ]
    
    # 디렉토리 존재 확인
    missing_dirs = []
    dataset_info = {}
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            # 파일 수 계산
            files = list(Path(dir_path).glob('*.*'))
            dataset_info[dir_path] = len(files)
    
    if missing_dirs:
        print("Warning: Following directories are missing:")
        for dir_path in missing_dirs:
            print(f"- {dir_path}")
        return False
    
    # 데이터셋 정보 출력
    print("\nDataset Structure:")
    print(f"Training Images: {dataset_info['data/train/images']}")
    print(f"Training Labels: {dataset_info['data/train/labels']}")
    print(f"Validation Images: {dataset_info['data/val/images']}")
    print(f"Validation Labels: {dataset_info['data/val/labels']}")
    print(f"Test Images: {dataset_info['data/test/images']}")
    print(f"Test Labels: {dataset_info['data/test/labels']}")
    
    # 이미지와 레이블 수 일치 확인
    if dataset_info['data/train/images'] != dataset_info['data/train/labels']:
        print("\nWarning: Number of training images and labels don't match!")
    if dataset_info['data/val/images'] != dataset_info['data/val/labels']:
        print("\nWarning: Number of validation images and labels don't match!")
    if dataset_info['data/test/images'] != dataset_info['data/test/labels']:
        print("\nWarning: Number of test images and labels don't match!")
    
    return True

if __name__ == "__main__":
    print("Starting dental plaque detection model training...")
    
    # 데이터셋 구조 검증
    if not validate_dataset_structure():
        print("Please run prepare_data.py first to set up the dataset structure")
        exit(1)
    
    # 기존 dataset.yaml 파일 사용
    data_yaml = 'data/dataset.yaml'
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found. Please check the dataset preparation.")
        exit(1)
    
    # 모델 학습
    model, results = train_model(
        data_yaml_path=data_yaml,
        epochs=100,
        img_size=640,
        batch_size=8,  # GPU 메모리에 따라 조정
        workers=4      # CPU 코어 수에 따라 조정
    )
    
    if model is not None:
        # 검증 실행
        print("\nRunning final validation...")
        val_results = model.val(data=data_yaml)
        
        print("\nTraining completed successfully!")
        print("Model saved at: models/best.pt")
        print("\nValidation metrics:")
        print(f"Segmentation mAP50: {val_results.seg.map50:.3f}")
        print(f"Segmentation mAP50-95: {val_results.seg.map:.3f}")
        
        # 추가 메트릭 출력
        print("\nDetailed Metrics:")
        print(f"Precision: {val_results.seg.precision:.3f}")
        print(f"Recall: {val_results.seg.recall:.3f}")
        print(f"F1-Score: {val_results.seg.f1:.3f}")
    else:
        print("\nTraining failed. Please check the error messages above.") 