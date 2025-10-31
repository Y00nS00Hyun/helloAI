# Fake News Detection Competition

가짜뉴스 판별 대회 프로젝트

## ⚠️ 중요: 대회 제약사항 준수

- ✅ **사전학습 모델 절대 사용 금지** - 모든 모델은 처음부터 학습
- ✅ **외부 API 사용 금지** - OpenAI, Google API 등 사용 불가
- ✅ **외부 데이터셋 사용 금지** - 제공된 데이터셋만 사용
- ✅ **모델 크기**: 최대 10GB 이하
- ✅ **추론 시간**: 30분 이내

## 프로젝트 구조

```
.
├── data/                    # 데이터셋 폴더
│   ├── dataset_1.csv
│   ├── dataset_2.csv
│   ├── dataset_3.csv
│   └── readme.txt
├── models/                  # 학습된 모델 저장
│   └── best.pt
├── configs/                 # 하이퍼파라미터 설정
│   ├── bilstm.yaml
│   └── transformer.yaml
├── model_definitions/       # 모델 정의
│   ├── bilstm.py           # BiLSTM 모델 (사전학습 없음)
│   └── transformer.py      # CNN 모델 (Transformer 대신)
├── utils/                   # 유틸리티 함수
│   ├── data_loader.py
│   ├── preprocessing.py    # 간단한 토크나이저 (사전학습 없음)
│   └── metrics.py
├── baseline.ipynb          # End-to-end 학습 노트북
├── train.py                # 학습 스크립트
├── api_server.py           # FastAPI 서버
└── requirements.txt
```

## 사용법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. API Key 설정
API 서버를 실행하기 전에 환경변수에 API Key를 설정하세요:
```bash
# Linux/Mac
export API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:API_KEY="your-api-key-here"

# Windows (CMD)
set API_KEY=your-api-key-here
```

또는 `api_server.py` 파일에서 직접 수정할 수도 있습니다:
```python
API_KEY = os.getenv("API_KEY", "your-api-key-here")
```

### 3. 학습
```bash
# BiLSTM 모델 학습
python train.py --model bilstm

# CNN 모델 학습 (transformer 옵션 사용)
python train.py --model transformer

# 설정 파일 사용
python train.py --model bilstm --config configs/bilstm.yaml
```

### 4. API 서버 실행
```bash
python api_server.py
```

서버는 `http://0.0.0.0:8000`에서 실행됩니다.

### 5. API 사용
모든 POST 엔드포인트는 `X-API-Key` 헤더가 필요합니다:

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}

# 단일 텍스트 추론
response = requests.post(
    "http://localhost:8000/infer",
    json={"text": "뉴스 텍스트..."},
    headers=headers
)
```

## API 엔드포인트

- `POST /infer`: 단일 텍스트 추론 (X-API-Key 필요)
- `POST /infer_csv`: CSV 파일 일괄 추론 (X-API-Key 필요)
- `POST /reload_model`: 모델 다시 불러오기 (X-API-Key 필요)
- `POST /validate`: 검증 데이터 평가 (X-API-Key 필요)
- `GET /health`: 헬스 체크 (API Key 불필요)
- `GET /`: 루트 엔드포인트 (API Key 불필요)

## Swagger UI

API 문서 및 테스트:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 모델 정보

### BiLSTM 모델
- 순수 PyTorch 구현
- Embedding + BiLSTM + 분류기
- 사전학습 없음

### CNN 모델 (Transformer 옵션)
- 순수 PyTorch 구현
- Embedding + CNN + 분류기
- 사전학습 없음
- 여러 크기의 필터 사용 (3, 4, 5)

## 주의사항

1. **사전학습 모델 사용 금지**: 모든 모델은 처음부터 학습합니다.
2. **토크나이저**: 학습 데이터로부터 어휘를 구축합니다.
3. **API Key**: 모든 POST 엔드포인트는 API Key가 필요합니다.
4. **모델 저장**: 학습 완료 후 `models/best.pt`로 자동 저장됩니다.
