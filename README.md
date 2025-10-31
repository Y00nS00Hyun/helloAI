# Fake News Detection Competition

가짜뉴스 판별 대회 프로젝트

## ⚠️ 중요: 대회 제약사항 준수

- ✅ **사전학습 모델 절대 사용 금지** - 모든 모델은 처음부터 학습
- ✅ **외부 API 사용 금지** - OpenAI, Google API 등 사용 불가
- ✅ **외부 데이터셋 사용 금지** - 제공된 데이터셋만 사용
- ✅ **모델 크기**: 최대 10GB 이하
- ✅ **추론 시간**: 30분 이내
- ✅ **평가 지표**: Macro F1 Score (Fake=1 positive, Real=0 negative)

## 빠른 시작

### 1. 데이터 분할
```bash
python utils/split_data.py --ratio 0.8
# 생성: data/train.csv, data/validation.csv
```

### 2. 학습
```bash
# BiLSTM 모델 학습
python train.py --model bilstm --device cuda

# CNN 모델 학습 (C. 모델명 명확화)
python train.py --model cnn --device cuda
# 또는 호환성을 위해 transformer도 가능
python train.py --model transformer --device cuda
```

### 3. API 서버 실행
```bash
# API Key 설정 (환경변수)
export API_KEY="your-api-key-here"  # Linux/Mac
# 또는 Windows PowerShell: $env:API_KEY="your-api-key-here"

python api_server.py
```

## 프로젝트 구조

```
.
├── data/                    # 데이터셋 폴더
│   ├── dataset_1.csv        # 원본 데이터
│   ├── dataset_2.csv        # 원본 데이터
│   ├── dataset_3.csv        # 원본 데이터
│   ├── train.csv           # 분할 후 생성 (split_data.py)
│   └── validation.csv      # 분할 후 생성 (split_data.py)
├── models/                  # 학습된 모델 저장
│   └── best.pt             # 제출 대상 모델 파일
├── configs/                 # 하이퍼파라미터 설정
│   ├── bilstm.yaml
│   └── transformer.yaml
├── model_definitions/       # 모델 정의
│   ├── __init__.py         # MODEL_REGISTRY 포함
│   ├── bilstm.py           # BiLSTM 모델 (사전학습 없음)
│   └── transformer.py      # CNN 모델 (Transformer 대신)
├── utils/                   # 유틸리티 함수
│   ├── split_data.py       # 데이터 분할 스크립트 ⭐
│   ├── data_loader.py      # 데이터 로딩
│   ├── preprocessing.py    # 토크나이저 (사전학습 없음)
│   └── metrics.py          # Macro F1 Score 평가
├── baseline.ipynb          # End-to-end 학습 노트북
├── train.py                # 학습 스크립트
├── api_server.py           # FastAPI 서버
├── requirements.txt
└── PERFORMANCE_TIPS.md     # 성능 개선 가이드
```

## 사용법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 분할 (필수)
먼저 원본 데이터를 train/validation으로 분할해야 합니다:
```bash
python utils/split_data.py --ratio 0.8
```

옵션:
- `--ratio`: 학습 데이터 비율 (기본값: 0.8)
- `--data_dir`: 데이터 디렉토리 (기본값: data)
- `--random_state`: 랜덤 시드 (기본값: 42)

### 3. 학습
```bash
# 기본 학습
python train.py --model bilstm --device cuda

# 설정 파일 사용
python train.py --model bilstm --config configs/bilstm.yaml --device cuda
```

### 4. API 서버 실행
```bash
# API Key 설정
export API_KEY="your-api-key-here"

# 서버 실행
python api_server.py
```

서버는 `http://0.0.0.0:8000`에서 실행됩니다.

### 5. API 사용
모든 POST 엔드포인트는 `X-API-Key` 헤더가 필요합니다:

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}

# 모델 리로드
requests.post("http://localhost:8000/reload_model", headers=headers)

# 단일 텍스트 추론 (A. title+text 지원)
response = requests.post(
    "http://localhost:8000/infer",
    json={
        "text": "뉴스 본문...",
        "title": "뉴스 제목"  # 선택사항
    },
    headers=headers
)

# 검증 데이터 평가 (최적 임계값 자동 계산)
response = requests.post(
    "http://localhost:8000/validate",
    headers=headers
)
# 응답에 optimal_threshold 포함됨
```

## API 엔드포인트

- `POST /infer`: 단일 텍스트 추론 (title+text 지원, X-API-Key 필요)
- `POST /infer_csv`: CSV 파일 일괄 추론 (title 자동 탐지, only_prediction 옵션, X-API-Key 필요)
- `POST /reload_model`: 모델 다시 불러오기 (X-API-Key 필요)
- `POST /validate`: 검증 데이터 평가 - **Macro F1 Score 및 최적 임계값 반환** (X-API-Key 필요)
- `GET /health`: 헬스 체크 (API Key 불필요)
- `GET /`: 루트 엔드포인트 (API Key 불필요)

## Swagger UI

API 문서 및 테스트:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Swagger UI에서 테스트 시 상단 `Authorize` 버튼을 클릭하여 API Key를 먼저 입력하세요!**

## 모델 정보

### BiLSTM 모델
- 순수 PyTorch 구현
- Embedding + BiLSTM + 분류기
- 사전학습 없음
- Attention mask를 이용한 평균 풀링

### CNN 모델 (Transformer 옵션)
- 순수 PyTorch 구현
- Embedding + CNN 필터 (3, 4, 5) + 분류기
- 사전학습 없음

### 새 모델 추가하기
1. `model_definitions/my_model.py` 파일 생성
2. `model_definitions/__init__.py`의 `MODEL_REGISTRY`에 등록:
```python
from model_definitions.my_model import MyModel

MODEL_REGISTRY = {
    "bilstm": BiLSTMModel,
    "transformer": TransformerModel,
    "my_model": MyModel,  # 추가
}
```
3. 학습: `python train.py --model my_model --device cuda`

## 평가 지표

대회 평가 기준: **Macro F1 Score**

```
F1_pos = 2TP / (2TP + FP + FN)  # Fake 클래스 F1
F1_neg = 2TN / (2TN + FP + FN)  # Real 클래스 F1
Macro F1 = (F1_pos + F1_neg) / 2
```

라벨 매핑:
- **Real = 0** (negative)
- **Fake = 1** (positive)

## 주의사항

1. **데이터 분할 필수**: 학습 전에 `utils/split_data.py`를 실행해야 합니다.
2. **사전학습 모델 사용 금지**: 모든 모델은 처음부터 학습합니다.
3. **토크나이저**: 학습 데이터로부터 어휘를 구축합니다.
4. **API Key**: 모든 POST 엔드포인트는 API Key가 필요합니다.
5. **모델 저장**: 학습 완료 후 `models/best.pt`로 자동 저장됩니다.
6. **모델 리로드**: API 서버 사용 전 `/reload_model` 호출 필요합니다.

## 성능 개선

자세한 성능 개선 방법은 `PERFORMANCE_TIPS.md`를 참고하세요.

주요 개선 방안:
- 학습률 스케줄러 적용
- 클래스 가중치 적용 (불균형 데이터)
- 하이퍼파라미터 튜닝
- 전처리 강화
- 앙상블 모델

## 체크리스트

학습 전 확인:
- [ ] `utils/split_data.py` 실행 완료
- [ ] `data/train.csv`, `data/validation.csv` 존재 확인
- [ ] API Key 설정 확인
- [ ] 모델 레지스트리 등록 확인 (새 모델 사용 시)

제출 전 확인:
- [ ] `models/best.pt` 파일 존재 확인
- [ ] `/reload_model`로 모델 로드 확인
- [ ] `/validate`로 Macro F1 Score 확인
- [ ] 모델 크기 10GB 이하 확인
- [ ] 추론 시간 30분 이내 확인
