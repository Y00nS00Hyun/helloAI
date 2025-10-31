# 성능 개선 가이드

## 현재 상태

✅ **참가자 가이드 요구사항 충족:**
- 데이터 분할 스크립트 (`utils/split_data.py`)
- 모델 레지스트리 (`MODEL_REGISTRY`)
- Macro F1 Score 평가
- 라벨 매핑 (Real=0, Fake=1)
- API Key 인증
- 사전학습 모델 미사용

## 성능 개선 방법

### 1. 전처리 개선

#### 현재 전처리 (`utils/preprocessing.py`)
- 소문자 변환
- 특수문자 제거
- 공백 정규화

#### 개선 방안
```python
# 불용어 제거 (한국어/영어)
# 숫자 정규화 (모든 숫자를 <NUM> 토큰으로)
# URL 제거
# 이모지 제거
# 반복 문자 정규화 (예: "좋아요요요" → "좋아요")
```

### 2. 토크나이저 개선

#### 현재: 단어 기반 토크나이저
- 최소 빈도 2 이상인 단어만 사용
- 어휘 크기 30,000

#### 개선 방안
```python
# 서브워드 토크나이저 구현 (BPE 유사)
# 문자 레벨 토크나이저 시도
# 어휘 크기 조정 (데이터 크기에 따라)
# 최소 빈도 조정 (1로 낮추면 더 많은 단어 포함)
```

### 3. 모델 아키텍처 개선

#### BiLSTM 모델
```yaml
# configs/bilstm.yaml 수정 제안
model:
  embedding_dim: 300 → 256  # 메모리 절약
  hidden_dim: 256 → 512     # 성능 향상
  num_layers: 2 → 3        # 깊이 증가
  dropout: 0.3 → 0.5       # 정규화 강화
```

#### CNN 모델
```yaml
# configs/transformer.yaml 수정 제안
model:
  embedding_dim: 300 → 256
  num_filters: 100 → 128    # 필터 수 증가
  filter_sizes: [3, 4, 5] → [2, 3, 4, 5]  # 더 다양한 필터
```

#### 새로운 모델 시도
- **GRU 모델**: LSTM보다 빠르고 메모리 효율적
- **Attention 메커니즘**: LSTM 출력에 어텐션 추가
- **앙상블**: 여러 모델 조합

### 4. 학습 전략 개선

#### 하이퍼파라미터 튜닝
```yaml
training:
  batch_size: 32 → 64 (GPU 메모리 여유 시)
  learning_rate: 0.001 → 0.0005 (더 안정적인 학습)
  num_epochs: 10 → 20 (더 많은 학습)
  
  # 학습률 스케줄러
  scheduler: cosine  # cosine annealing 사용
  
  # 조기 종료
  early_stopping:
    patience: 5 → 7  # 더 많은 기회
    min_delta: 0.001 → 0.0001  # 더 작은 개선도 저장
```

#### 손실 함수 개선
```python
# 클래스 불균형 고려
from torch.nn import CrossEntropyLoss

# 클래스별 가중치 적용
class_weights = torch.tensor([1.0, 2.0])  # Fake에 더 큰 가중치
criterion = CrossEntropyLoss(weight=class_weights)
```

### 5. 데이터 증강

```python
# 텍스트 증강 기법 (사전학습 모델 없이)
# 1. 동의어 교체 (사전 기반)
# 2. 랜덤 삭제 (5-10% 단어 삭제)
# 3. 랜덤 교환 (단어 순서 변경)
# 4. 백트랜슬레이션 (번역 후 재번역)
```

### 6. 교차 검증

```python
# K-Fold 교차 검증으로 더 안정적인 평가
from sklearn.model_selection import KFold

# 5-fold 교차 검증으로 모델 선택
```

### 7. 앙상블

```python
# 여러 모델의 예측을 평균
predictions = (model1_pred + model2_pred + model3_pred) / 3
```

## 우선순위별 개선 체크리스트

### 🟢 높은 우선순위 (즉시 적용 가능)
- [ ] **Macro F1 Score 기준으로 모델 선택** (이미 적용됨)
- [ ] **학습률 스케줄러 적용** (cosine annealing)
- [ ] **클래스 가중치 적용** (불균형 데이터)
- [ ] **어휘 크기 및 최소 빈도 조정**

### 🟡 중간 우선순위 (시간 여유 시)
- [ ] **전처리 강화** (불용어, 숫자 정규화)
- [ ] **하이퍼파라미터 튜닝** (그리드 서치)
- [ ] **앙상블 모델 학습**

### 🔵 낮은 우선순위 (최종 최적화)
- [ ] **교차 검증**
- [ ] **데이터 증강**
- [ ] **새로운 모델 아키텍처 실험**

## 제약사항 고려

### 모델 크기 ≤ 10GB
- 임베딩 차원, 히든 차원 조정으로 크기 조절
- 모델 파라미터 수 계산: `sum(p.numel() for p in model.parameters())`

### 추론 시간 ≤ 30분
- 배치 크기 증가로 속도 향상
- 모델 단순화 (레이어 수 감소)

## 모니터링

학습 중 다음 지표 확인:
- **Macro F1 Score** (대회 평가 기준)
- **F1 Real vs F1 Fake** (불균형 확인)
- **Train Loss vs Val Loss** (과적합 확인)
- **Confusion Matrix** (어떤 클래스에서 실패하는지)

## 예상 성능 향상

현재 기준선에서:
- 기본 개선 (스케줄러, 가중치): +2-3%
- 전처리 개선: +1-2%
- 하이퍼파라미터 튜닝: +2-4%
- 앙상블: +1-3%

**총 예상 개선: +6-12% Macro F1 Score**

