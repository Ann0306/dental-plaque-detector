# 치아 치석 감지 시스템

이 프로젝트는 YOLOv8 세그멘테이션 모델을 사용하여 치아 사진에서 치석을 감지하고 분석하는 웹 애플리케이션입니다. 사용자가 치아 사진을 업로드하면 AI가 치아와 치석 영역을 감지하고, 치아 대비 치석의 비율을 계산합니다.

## 주요 기능

- 치아 사진 업로드 및 분석
- 치아와 치석 영역 세그멘테이션
- 치아 면적 대비 치석 면적의 비율 계산
- 세그멘테이션된 결과 시각화

## 시스템 구성

이 프로젝트는 다음과 같은 구성 요소로 이루어져 있습니다:

### 백엔드

- **Flask**: 웹 서버 및 API
- **YOLOv8**: 치아와 치석 감지를 위한 세그멘테이션 모델
- **OpenCV**: 이미지 처리 및 시각화

### 프론트엔드

- **HTML/CSS/JavaScript**: 사용자 인터페이스
- **Bootstrap**: 반응형 UI 컴포넌트

## 설치 및 실행 방법

### 요구 사항

- Python 3.10 이상
- CUDA 지원 그래픽 카드 (권장, 없어도 실행 가능)

### 설치 과정

1. 저장소 클론
   ```
   git clone <repository-url>
   cd dental_plaque_app
   ```

2. 가상 환경 생성 및 활성화
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. 의존성 설치
   ```
   pip install -r requirements.txt
   ```

### 실행 방법

1. 서버 실행
   ```
   python app_final.py
   ```

2. 웹 브라우저에서 접속
   ```
   http://localhost:5000
   ```

## 데이터 준비 및 모델 학습

이 프로젝트는 치아와 치석이 레이블링된 커스텀 데이터셋을 사용하여 YOLOv8 모델을 파인튜닝합니다.

### 데이터 준비

1. 원본 데이터셋을 `raw_data` 폴더에 위치시킵니다.
   - `raw_data/images`: 원본 치아 이미지
   - `raw_data/masks`: 각 이미지에 해당하는 마스크 이미지 (치아: 1, 치석: 2)

2. 데이터 전처리 스크립트 실행
   ```
   python prepare_data.py
   ```
   이 스크립트는 원본 데이터를 YOLO 형식으로 변환하고, 학습/검증/테스트 세트로 분할합니다.

### 모델 학습

1. 모델 학습 스크립트 실행
   ```
   python train_model.py
   ```
   이 스크립트는 YOLOv8 모델을 파인튜닝하고, 최적의 가중치를 `models/best.pt`에 저장합니다.

## 사용된 패키지 버전

프로젝트는 다음 주요 패키지 버전을 사용합니다:
- Flask 2.0.1
- Werkzeug 2.0.1
- Ultralytics 8.0.0 (YOLOv8)
- OpenCV Python 4.7.0.72
- PyTorch 2.0.0
- TorchVision 0.15.0

## 프로젝트 구조

```
dental_plaque_app/
│
├── app_final.py            # 메인 Flask 애플리케이션
├── train_model.py          # 모델 학습 스크립트
├── prepare_data.py         # 데이터 준비 스크립트
├── requirements.txt        # 의존성 목록
│
├── data/                   # 학습 데이터셋
│   ├── dataset.yaml        # 데이터셋 설정 파일
│   ├── train/              # 학습 데이터
│   ├── val/                # 검증 데이터
│   └── test/               # 테스트 데이터
│
├── models/                 # 학습된 모델 저장 위치
│   └── best.pt             # 최적의 가중치
│
├── static/                 # 정적 파일
│   └── uploads/            # 업로드된 이미지 및 결과
│
└── templates/              # HTML 템플릿
    ├── index.html          # 메인 페이지
    └── result.html         # 결과 페이지
```

## 이미지 크기 처리

웹 애플리케이션은 다양한 크기의 이미지를 처리할 수 있도록 설계되었습니다. YOLOv8 모델은 입력 이미지를 384x640 크기로 처리하며, 결과 마스크는 원본 이미지 크기로 리사이징되어 정확한 면적 계산과 시각화를 제공합니다.

## 참고 자료

- [YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [OpenCV 공식 문서](https://docs.opencv.org/)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요. 

# Ensure these values sum exactly to 1.0
split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1} 