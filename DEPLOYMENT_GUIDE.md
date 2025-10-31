# code-server 배포 가이드

code-server에는 대회 조직자가 제공한 **기본 템플릿 코드와 데이터**가 있습니다.
여러분의 **리팩토링된 코드**를 서버에 올리는 방법을 안내합니다.

## 📁 code-server에 있는 파일들

대회 서버에는 보통 다음과 같은 파일들이 있습니다:

```
/
├── data/
│   ├── dataset_1.csv          # 원본 데이터 (중요!)
│   ├── dataset_2.csv          # 원본 데이터 (중요!)
│   ├── dataset_3.csv          # 원본 데이터 (중요!)
│   └── readme.txt             # 데이터 설명
├── utils/
│   ├── split_data.py          # 데이터 분할 스크립트 (제공본)
│   ├── dataset.py             # 데이터 로더 (제공본, 선택사항)
│   └── vocab.py               # 토크나이저 (제공본, 선택사항)
├── model_definitions/
│   └── ...                    # 기본 모델 정의 (제공본)
├── configs/
│   └── ...                    # 기본 설정 파일
├── baseline.ipynb             # 예제 노트북 (참고용)
├── train.py                   # 기본 학습 스크립트 (제공본)
├── api_server.py              # 기본 API 서버 (제공본)
└── requirements.txt           # 의존성 목록
```

## 🎯 통합 전략

### 방법 1: 로컬 코드 전체 업로드 (추천)

여러분의 리팩토링된 코드를 그대로 사용하는 방법입니다.

#### 단계 1: 로컬 코드 압축
```bash
# Windows (PowerShell)
Compress-Archive -Path . -DestinationPath code.zip -Exclude "*.pt", "__pycache__", ".git"

# 또는 수동으로 선택:
# - api_server.py
# - train.py
# - utils/ (전체 폴더)
# - model_definitions/ (전체 폴더)
# - configs/ (전체 폴더)
# - requirements.txt
# - README.md
```

#### 단계 2: code-server에 업로드
1. code-server 웹 IDE에서 **파일 탐색기** 열기
2. 압축 해제하거나 파일 직접 업로드
3. **중요**: `data/` 폴더는 **건드리지 말 것** (원본 데이터 유지)

#### 단계 3: 의존성 설치
code-server의 터미널에서:
```bash
pip install -r requirements.txt
```

### 방법 2: 파일별 선택적 업로드

필요한 파일만 골라서 업로드:

#### 필수 업로드 파일
✅ **반드시 업로드해야 할 파일:**
- `train.py` (리팩토링된 버전)
- `api_server.py` (리팩토링된 버전)
- `utils/` 폴더 전체 (리팩토링된 버전)
- `model_definitions/` 폴더 전체
- `configs/` 폴더 전체
- `requirements.txt`

#### 선택적 파일
⚠️ **선택사항 (있으면 좋지만 필수 아님):**
- `README.md`
- `PERFORMANCE_TIPS.md`
- `.gitignore`

#### 건드리지 말 것
❌ **서버에 그대로 둘 것:**
- `data/` 폴더 (원본 데이터 보존)
- 대회에서 제공한 다른 중요한 파일들

## 🔄 파일 비교 및 통합

### utils/split_data.py 비교

**서버 버전 vs 로컬 버전:**
- 서버의 `utils/split_data.py`는 대회에서 제공한 기본 버전
- 로컬의 `utils/split_data.py`는 우리가 만든 버전 (동일한 기능)
- **통합 방법**: 로컬 버전으로 교체 (기능 동일하지만 코드가 더 깔끔)

### baseline.ipynb

- 서버에 있는 `baseline.ipynb`는 대회에서 제공한 예제 노트북
- **선택사항**: 그대로 두거나, 로컬에 있는 것이 있다면 교체 가능
- 주요 목적: 전체 흐름 이해용 참고 자료

## 📋 통합 체크리스트

업로드 전 확인:

- [ ] `data/` 폴더는 건드리지 않음
- [ ] `utils/split_data.py` 업로드 완료
- [ ] `utils/data_loader.py` 업로드 완료
- [ ] `utils/preprocessing.py` 업로드 완료
- [ ] `utils/metrics.py` 업로드 완료
- [ ] `utils/model_utils.py` 업로드 완료
- [ ] `utils/config_utils.py` 업로드 완료
- [ ] `utils/constants.py` 업로드 완료
- [ ] `utils/__init__.py` 업로드 완료
- [ ] `model_definitions/` 폴더 전체 업로드
- [ ] `configs/` 폴더 전체 업로드
- [ ] `train.py` 업로드 완료
- [ ] `api_server.py` 업로드 완료
- [ ] `requirements.txt` 업로드 완료

## 🚀 업로드 후 작업

### 1. 데이터 분할 실행
```bash
python utils/split_data.py --ratio 0.8
```

### 2. 학습 실행
```bash
python train.py --model bilstm --device cuda
```

### 3. API 서버 실행
```bash
# API Key 설정
export API_KEY="your-api-key-here"

python api_server.py
```

## ⚠️ 주의사항

1. **데이터 백업**: `data/` 폴더의 원본 파일은 절대 삭제/수정하지 마세요
2. **경로 확인**: code-server의 작업 루트가 `/` 인지 확인
3. **권한 확인**: 파일 업로드 권한이 있는지 확인
4. **버전 충돌**: 기존 파일과 이름이 같은 경우 덮어쓰기 확인

## 🔍 문제 해결

### 문제 1: 업로드된 파일이 보이지 않음
- 브라우저 새로고침 (F5)
- 파일 탐색기 새로고침
- 파일이 올바른 디렉토리에 있는지 확인

### 문제 2: import 에러 발생
- `utils/__init__.py`가 제대로 업로드되었는지 확인
- 모든 `utils/` 하위 파일이 업로드되었는지 확인
- `requirements.txt` 의존성 설치 확인

### 문제 3: 모델이 로드되지 않음
- `models/best.pt` 파일 경로 확인
- 모델 타입과 설정 파일 일치 확인
- 어휘 크기가 학습 시와 동일한지 확인

## 💡 빠른 업로드 스크립트 (선택사항)

code-server 터미널에서 직접 파일을 수정할 수도 있습니다:

1. **파일 직접 편집**: code-server 웹 IDE에서 파일 열고 수정
2. **Git 사용**: code-server에서 Git이 설정되어 있다면 pull 사용
3. **SCP/SFTP**: 서버에 SSH 접근 권한이 있다면 직접 전송

## 📝 최종 확인

업로드 후 다음 명령어로 확인:

```bash
# 파일 구조 확인
ls -la
ls utils/
ls model_definitions/

# Python 경로 확인
python -c "import sys; print(sys.path)"

# 모듈 import 테스트
python -c "from utils import load_datasets; print('OK')"
python -c "from model_definitions import MODEL_REGISTRY; print(list(MODEL_REGISTRY.keys()))"
```

---

**결론**: 로컬의 리팩토링된 코드를 그대로 code-server에 업로드하고, `data/` 폴더만 그대로 유지하면 됩니다!

