import os
import sys
import cv2
import numpy as np

# Patch for the Git issue - apply before importing ultralytics
import subprocess
from pathlib import Path

# Monkey patch subprocess.run to handle Git commands
original_run = subprocess.run
def patched_run(args, *argsp, **kwargs):
    # Check if this is a git command
    if isinstance(args, list) and args and args[0] == 'git':
        print("Git command intercepted:", args)
        # Create a dummy successful result
        if args[1:] and args[1] == 'rev-parse':
            class DummyResult:
                returncode = 0
                stdout = b''
            return DummyResult()
        # For any other git command
        raise FileNotFoundError("Git command intercepted and blocked for safety")
    # For non-git commands, use the original function
    return original_run(args, *argsp, **kwargs)

# Apply the patch
subprocess.run = patched_run

# Patch Path class to handle get_git_root_dir
original_path = Path

# Now we can import Flask and Ultralytics safely
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
from datetime import datetime

app = Flask(__name__)

# 설정
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('models', 'best.pt')  # 파인튜닝된 모델 경로

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 최대 16MB 업로드

# 모델 로드 (서버 시작시 한 번만 로드)
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")
    print("Will attempt to use a pretrained YOLO model instead.")
    model = YOLO('yolov8n-seg.pt')  # 기본 모델 사용

# 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 치아 및 치석 면적 계산 함수
def calculate_areas(masks, classes):
    # 치아(class 0)와 치석(class 1)의 면적 계산
    tooth_area = 0
    plaque_area = 0
    
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            if class_id == 0:  # 치아 클래스
                tooth_area += np.sum(mask)
            elif class_id == 1:  # 치석 클래스
                plaque_area += np.sum(mask)
    
    # 치석 비율 계산 (치아 면적이 0인 경우 예외 처리)
    if tooth_area > 0:
        plaque_ratio = (plaque_area / tooth_area) * 100
    else:
        plaque_ratio = 0
    
    return tooth_area, plaque_area, plaque_ratio

# 세그멘테이션 마스크 시각화 함수
def visualize_segmentation(image, masks, classes):
    if masks is None or len(masks) == 0:
        return image, None
    
    # 원본 이미지 복사
    result_img = image.copy()
    mask_img = np.zeros_like(image)
    
    # 클래스별 색상 설정 (BGR 형식)
    colors = {
        0: (0, 255, 0),    # 치아: 초록색
        1: (0, 0, 255)     # 치석: 빨간색
    }
    
    # 각 마스크 적용
    for i, mask in enumerate(masks):
        class_id = int(classes[i])
        color = colors.get(class_id, (255, 255, 255))  # 기본값: 흰색
        
        # 마스크를 3채널로 확장하여 색상 적용
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color
        
        # 마스크 이미지에 누적
        mask_img = cv2.addWeighted(mask_img, 1, colored_mask, 0.5, 0)
        
        # 결과 이미지에 마스크 오버레이
        result_img = cv2.addWeighted(result_img, 1, colored_mask, 0.5, 0)
    
    return result_img, mask_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 파일명 안전하게 저장
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 이미지 처리 및 모델 추론
        results = process_image(filepath)
        
        return render_template('result.html', results=results)
    
    return redirect(request.url)

def process_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    
    if image is None:
        return {"error": "Failed to load image. File may be corrupt or in an unsupported format."}
    
    # 원본 이미지 크기 저장
    original_height, original_width = image.shape[:2]
    
    try:
        # 모델 추론
        abs_path = os.path.abspath(image_path)
        results = model.predict(source=abs_path, conf=0.25, save=False)
    except Exception as e:
        return {"error": f"Error during model inference: {str(e)}"}
    
    # 결과 처리
    result_data = {}
    result_data['original_image'] = os.path.basename(image_path)
    
    if len(results) > 0:
        result = results[0]
        
        # 마스크와 클래스 추출
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            try:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # 각 마스크를 원본 이미지 크기로 리사이즈
                resized_masks = []
                for mask in masks:
                    # 마스크를 float32로 변환
                    mask = mask.astype(np.float32)
                    # 리사이즈
                    resized_mask = cv2.resize(mask, (original_width, original_height))
                    # 이진화 (0.5 임계값 사용)
                    resized_mask = (resized_mask > 0.5).astype(np.uint8)
                    resized_masks.append(resized_mask)
                
                # 면적 계산
                tooth_area, plaque_area, plaque_ratio = calculate_areas(resized_masks, classes)
                result_data['tooth_area'] = tooth_area
                result_data['plaque_area'] = plaque_area
                result_data['plaque_ratio'] = round(plaque_ratio, 2)
                
                # 세그먼테이션 결과 시각화
                result_image, mask_image = visualize_segmentation(image, resized_masks, classes)
                
                # 결과 이미지 저장
                result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{os.path.basename(image_path)}")
                cv2.imwrite(result_image_path, result_image)
                result_data['result_image'] = os.path.basename(result_image_path)
                
                if mask_image is not None:
                    mask_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{os.path.basename(image_path)}")
                    cv2.imwrite(mask_image_path, mask_image)
                    result_data['mask_image'] = os.path.basename(mask_image_path)
            except Exception as e:
                result_data['error'] = f"Error processing segmentation results: {str(e)}"
        else:
            result_data['error'] = "No segmentation masks found in the results"
    else:
        result_data['error'] = "No detection results"
    
    return result_data

if __name__ == '__main__':
    # 업로드 폴더가 없으면 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 모델 폴더가 없으면 생성
    os.makedirs('models', exist_ok=True)
    
    # 디버그 모드로 서버 실행
    app.run(debug=True, host='0.0.0.0', port=5000) 