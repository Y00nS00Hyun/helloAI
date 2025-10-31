# 변경 사항 (리스크 보완)

## ✅ 구현 완료된 개선사항

### A. 입력 스키마 호환성 (필수) ✅
- `/infer` 엔드포인트가 `title`과 `text` 모두 받도록 개선
- `title`이 있으면 자동으로 `title + text` 형태로 결합
- `/infer_csv`에서도 `title` 컬럼 자동 탐지 및 결합
- `utils/data_loader.py`에 `combine_title_text()` 함수 추가

### B. 평가 지표/라벨 정의 일치 (필수) ✅
- 라벨 매핑 검증 강화: `/validate` 응답에 라벨 매핑 정보 포함
- Real=0, Fake=1 매핑 명확화 및 검증 로그 추가
- `utils/metrics.py`에서 라벨 매핑 검증 로그 출력

### C. 모델명 혼동 제거 (권장) ✅
- `MODEL_REGISTRY`에 `"cnn"` 추가 (기존 `"transformer"`는 호환성 유지)
- `configs/cnn.yaml` 파일 추가
- README에서 `--model cnn` 사용 권장

### D. 임계값 튜닝 (권장) ✅
- `find_optimal_threshold()` 함수 추가 (0.1~0.9 범위 탐색)
- 학습 중 최적 임계값 자동 탐색
- `/validate`에서 최적 임계값 계산 및 반환
- 메타데이터에 최적 임계값 저장
- 추론 시 최적 임계값 사용

### E. 클래스 불균형 대응 (권장) ✅
- `CrossEntropyLoss`에 `class_weight` 자동 계산 및 적용
- 클래스 분포 로깅
- 학습 시 가중치 정보 출력

### F. CSV 처리 견고화 (권장) ✅
- `read_csv_robust()` 함수: 구분자 자동 감지 (csv.Sniffer)
- 여러 인코딩 시도 (utf-8, utf-8-sig, cp949, latin-1)
- 따옴표 처리 (`engine='python'`)
- 청크 처리 (10k 라인 단위)로 메모리 보호
- `only_prediction` 옵션 지원
- 친절한 에러 메시지 (사용 가능한 컬럼 표시)

### G. 재현성 강화 (권장) ✅
- `set_seed()` 함수: 모든 시드 고정 (random, numpy, torch, cudnn)
- `--seed` 인자 추가 (기본값: 42)
- `cudnn.deterministic = True` 설정
- 학습 메타데이터 저장 (`models/metadata.json`)
  - 설정, 시드, 점수, 최적 임계값 포함
- 모델 로드 시 메타데이터에서 최적 임계값 복원

## 📝 주요 변경 파일

1. **api_server.py**
   - `TextRequest`: title 필드 추가 (Optional)
   - `combine_title_text()` 함수 추가
   - `predict_text()`: title 지원, threshold 적용
   - `read_csv_robust()`: 견고한 CSV 파싱
   - `/infer_csv`: title 자동 탐지, 청크 처리, only_prediction 옵션
   - `/validate`: 라벨 매핑 정보, 최적 임계값 반환

2. **train.py**
   - `set_seed()`: 재현성 강화
   - `validate()`: 확률 배열 반환 옵션 추가
   - 클래스 가중치 자동 계산 및 적용
   - 최적 임계값 탐색 및 저장
   - `save_metadata()`: 메타데이터 저장

3. **utils/metrics.py**
   - `find_optimal_threshold()`: 최적 임계값 탐색 함수 추가
   - 라벨 매핑 검증 로그 추가

4. **utils/data_loader.py**
   - `combine_title_text()`: title+text 결합 함수 추가

5. **model_definitions/__init__.py**
   - `"cnn"` 키 추가 (MODEL_REGISTRY)

6. **configs/cnn.yaml**
   - 새로운 설정 파일 추가

## 🎯 사용 예시

### title+text 지원 (A)
```python
# /infer 요청
{
    "title": "뉴스 제목",
    "text": "뉴스 본문..."
}
```

### 최적 임계값 사용 (D)
```python
# 학습 후 자동으로 최적 임계값이 저장되고 사용됨
# /validate에서 확인 가능
```

### 클래스 불균형 대응 (E)
```python
# 학습 시 자동으로 클래스 가중치 계산 및 적용
# 로그에서 확인:
# Class weights: Real=0.500, Fake=2.000
```

### 견고한 CSV 처리 (F)
```python
# 다양한 구분자, 인코딩 자동 감지
# title 컬럼이 있으면 자동으로 사용
```

## ⚠️ 주의사항

1. **라벨 매핑**: Real=0, Fake=1로 통일 (B)
2. **임계값**: 학습 후 자동으로 최적값이 저장되며, 추론 시 사용됨 (D)
3. **재현성**: `--seed` 인자로 시드 고정 가능 (G)
4. **모델명**: `cnn` 사용 권장, `transformer`는 호환성 유지 (C)

