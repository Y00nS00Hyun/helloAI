# 추가 개선사항 (V2)

## 🔧 세부 개선 완료

### A. 입력 스키마 호환성 - [SEP] 토큰 명시
- ✅ `combine_title_text()` 함수에서 구분자를 `" [SEP] "`로 명시
- ✅ `/infer`: title이 있으면 `"title [SEP] text"` 형태로 결합
- ✅ `/infer_csv`: 컬럼 자동 탐지 (대소문자 무시)
- ✅ 빈 행 자동 제거 및 카운트
- ✅ 친절한 에러 메시지 (사용 가능한 컬럼, 제거된 빈 행 수)

### B. 라벨/지표 일치 검증 가시화
- ✅ `/validate` 응답 강화:
  ```json
  {
    "macro_f1": 0.81,
    "pos_label": "fake(1)",
    "threshold": 0.47,
    "f1_fake": 0.82,
    "f1_real": 0.80,
    "class_metrics": {...},
    "selection_criterion": "macro_f1",
    "label_mapping": {...}
  }
  ```
- ✅ 학습 시 best.pt 선정 기준 명시 (`selection_criterion: "macro_f1"`)
- ✅ 메타데이터에 모델 선택 기준 저장

### C. 네이밍 혼선 제거
- ✅ `MODEL_REGISTRY`에 `"cnn"` 추가
- ✅ README에 명확히 표기: "CNN 모델 (실제 Transformer가 아닌 TextCNN)"
- ✅ `transformer`는 호환성 유지를 위해 유지

### D. Threshold 튜닝
- ✅ 이미 구현 완료
- ✅ 메타데이터 저장 및 로드 확인

### E. 불균형/중복 대응
- ✅ 클래스 불균형: `CrossEntropyLoss(weight=class_weights)` 적용
- ✅ 중복 제거 강화:
  - text 기준 중복 제거
  - 정규화된 텍스트 기준 중복 제거
  - 중복 제거 통계 로깅

### F. CSV 대량 추론 견고화
- ✅ 구분자 자동 감지 (`csv.Sniffer`)
- ✅ 여러 인코딩 시도
- ✅ `engine='python'`으로 따옴표 안전 처리
- ✅ 청크 처리 (10k 라인 단위)
- ✅ `only_prediction` 옵션 지원
- ✅ 빈 행 카운트 및 제거
- ✅ 친절한 에러 메시지

### G. 재현성/체크포인트
- ✅ `git_sha` 추가 (메타데이터에 저장)
- ✅ 시드 고정 (`set_seed()`)
- ✅ 메타데이터에 모든 정보 저장:
  - config
  - seed
  - score (macro_f1)
  - threshold
  - git_sha
  - selection_criterion

### H. 운영 안전장치
- ✅ `/reload_model` Rate limit (분당 2회)
- ✅ 모델 로드 실패 시 롤백 처리
- ✅ `/health` 공개 OK (API Key 불필요)
- ✅ 모든 POST 엔드포인트 Key 필수 확인

## 📋 주요 변경사항

### api_server.py
1. **`combine_title_text()` 호출**: [SEP] 토큰 사용
2. **Rate limiting**: `/reload_model`에 분당 2회 제한
3. **롤백 기능**: 모델 로드 실패 시 이전 모델로 복구
4. **에러 메시지 개선**: CSV 파싱 실패 시 상세 정보 제공
5. **`/validate` 응답 강화**: 라벨 매핑, 임계값, 클래스별 메트릭 명시

### train.py
1. **`get_git_sha()`**: Git SHA 추출 함수 추가
2. **`save_metadata()`**: git_sha, selection_criterion 추가
3. **중복 제거 강화**: 정규화된 텍스트 기준 중복 제거

### utils/data_loader.py
1. **`combine_title_text()`**: 구분자를 `" [SEP] "`로 변경
2. **중복 제거 강화**: 정규화 후 중복 제거

## 🎯 사용 예시

### [SEP] 토큰 사용 (A)
```python
# title과 text가 " [SEP] "로 결합됨
{
    "title": "뉴스 제목",
    "text": "뉴스 본문"
}
# → "뉴스 제목 [SEP] 뉴스 본문"
```

### /validate 응답 (B)
```json
{
    "macro_f1": 0.815,
    "pos_label": "fake(1)",
    "threshold": 0.47,
    "f1_fake": 0.82,
    "f1_real": 0.81,
    "selection_criterion": "macro_f1"
}
```

### Rate Limit (H)
```bash
# 분당 최대 2회만 /reload_model 호출 가능
# 초과 시 429 에러 반환
```

## ✅ 체크리스트

- [x] [SEP] 토큰 명시
- [x] 라벨 매핑 가시화
- [x] 모델 선택 기준 명시
- [x] git_sha 추가
- [x] 중복 제거 강화
- [x] Rate limit
- [x] 롤백 기능
- [x] 에러 메시지 개선

