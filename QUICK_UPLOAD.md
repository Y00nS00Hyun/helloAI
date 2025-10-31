# 빠른 업로드 가이드

code-server에 코드를 빠르게 업로드하는 방법

## 방법 1: 파일 복사 & 붙여넣기 (가장 간단)

### code-server 웹 IDE에서:

1. **새 파일 생성** 또는 **기존 파일 열기**
2. 로컬에서 파일 내용 복사
3. code-server에서 붙여넣기
4. 저장

### 업로드해야 할 파일 목록:

```
필수 파일들:
├── api_server.py
├── train.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── constants.py
│   ├── config_utils.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── model_utils.py
│   ├── preprocessing.py
│   └── split_data.py
├── model_definitions/
│   ├── __init__.py
│   ├── bilstm.py
│   └── transformer.py
└── configs/
    ├── bilstm.yaml
    └── transformer.yaml
```

## 방법 2: Git 사용 (code-server에 Git 설정된 경우)

### 로컬에서:
```bash
git add .
git commit -m "Refactored code for competition"
git push origin main
```

### code-server에서:
```bash
git pull origin main
```

## 방법 3: 파일 업로드 기능 사용

code-server 웹 IDE에는 파일 업로드 기능이 있을 수 있습니다:
1. 파일 탐색기에서 폴더 우클릭
2. "Upload Files" 선택
3. 파일 선택하여 업로드

## 빠른 체크리스트

업로드 후 바로 확인:

```bash
# 1. 파일 존재 확인
ls -la train.py api_server.py
ls -la utils/
ls -la model_definitions/

# 2. Python 모듈 테스트
python -c "from utils import load_datasets; print('✓ utils OK')"
python -c "from model_definitions import MODEL_REGISTRY; print('✓ models OK')"

# 3. 데이터 확인
ls -la data/
python utils/split_data.py --ratio 0.8

# 4. 학습 테스트 (빠른 확인용)
python train.py --model bilstm --device cpu --config configs/bilstm.yaml
```

## ⚡ 초간단 버전

**가장 빠른 방법:**

1. code-server 웹 IDE 열기
2. `train.py`, `api_server.py` 열기
3. 로컬 코드 복사 → 붙여넣기 → 저장
4. `utils/` 폴더 내 모든 파일 동일하게 복사
5. `model_definitions/` 폴더 내 모든 파일 동일하게 복사
6. `requirements.txt` 확인 및 설치

끝! 🎉

