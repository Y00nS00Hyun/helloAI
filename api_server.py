"""
FastAPI 서버 - 가짜뉴스 판별 API
"""
import logging
import os
import csv
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Security, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from utils.data_loader import (
    load_datasets,
    split_train_val,
    prepare_data_for_training
)
from utils.preprocessing import TextTokenizer
from utils.metrics import calculate_metrics, find_optimal_threshold
from utils.model_utils import create_model
from utils.config_utils import load_config
from utils.constants import DEFAULT_CONFIG
from model_definitions import MODEL_REGISTRY

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", version="1.0.0")

# API Key 인증 설정
API_KEY = os.getenv("API_KEY", "your-api-key-here")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    API Key 검증

    Args:
        api_key: API Key

    Returns:
        검증된 API Key

    Raises:
        HTTPException: API Key가 유효하지 않은 경우
    """
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key. Please provide valid X-API-Key header."
        )
    return api_key


# 전역 변수
model: Optional[torch.nn.Module] = None
tokenizer: Optional[TextTokenizer] = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config: Optional[Dict[str, Any]] = None
model_type: Optional[str] = None
optimal_threshold: float = 0.5  # 기본값, 학습 시 업데이트됨


# Pydantic 모델
class TextRequest(BaseModel):
    """단일 텍스트 추론 요청 (A. 입력 스키마 호환성)"""
    text: str = Field(..., description="뉴스 본문 텍스트 (필수)")
    title: Optional[str] = Field(None, description="뉴스 제목 (선택사항, 있으면 함께 사용)")


class TextResponse(BaseModel):
    """단일 텍스트 추론 응답"""
    prediction: str  # 'real' or 'fake'
    confidence: float
    probabilities: Dict[str, float]
    threshold: float = Field(..., description="사용된 분류 임계값")


def combine_title_text(title: Optional[str], text: str, sep: str = " ") -> str:
    """
    title과 text를 결합 (A. 입력 스키마 호환성)

    Args:
        title: 제목 (없으면 None)
        text: 본문
        sep: 구분자

    Returns:
        결합된 텍스트
    """
    if title and title.strip():
        return f"{title.strip()}{sep}{text.strip()}"
    return text.strip()


def load_model(
    model_path: str = "models/best.pt",
    model_type_param: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    모델 로드

    Args:
        model_path: 모델 파일 경로
        model_type_param: 모델 타입
        config: 설정 딕셔너리

    Raises:
        FileNotFoundError: 모델 파일이 없는 경우
        ValueError: 모델 타입이 잘못된 경우
    """
    global model, tokenizer, model_config, model_type, optimal_threshold

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 모델 타입 및 설정 확인
    if not model_type_param:
        model_type_param = "bilstm"  # 기본값

    if config is None:
        config = load_config(model_name=model_type_param)

    model_config = config
    model_type = model_type_param

    # 토크나이저 로드
    tokenizer_config = config.get('tokenizer', {})
    tokenizer = TextTokenizer(
        vocab_size=tokenizer_config.get('vocab_size', 30000),
        max_length=tokenizer_config.get('max_length', 512)
    )

    # 학습 데이터로 어휘 구축
    try:
        df = load_datasets(data_dir='data')
        if 'label' in df.columns:
            train_df, _ = split_train_val(df, val_ratio=0.2, random_state=42)
            train_texts, _ = prepare_data_for_training(train_df)
        else:
            texts, _ = prepare_data_for_training(df)
            train_texts = texts

        min_freq = tokenizer_config.get('min_freq', 2)
        tokenizer.build_vocab(train_texts, min_freq=min_freq)
        logger.info(
            f"Vocabulary built with {len(tokenizer.tokenizer.word_to_idx)} tokens"
        )
    except Exception as e:
        logger.warning(f"Could not build vocabulary: {e}")

    # 모델 생성
    vocab_size = len(tokenizer.tokenizer.word_to_idx) if hasattr(
        tokenizer.tokenizer, 'word_to_idx'
    ) and tokenizer.tokenizer.word_to_idx else config.get(
        'model', {}
    ).get('vocab_size', 30000)

    model = create_model(
        model_name=model_type_param,
        vocab_size=vocab_size,
        config=config,
        device=device
    )

    # 모델 가중치 로드
    model.load_state_dict(
        torch.load(model_path_obj, map_location=device)
    )
    model = model.to(device)
    model.eval()

    # 메타데이터에서 optimal_threshold 로드
    metadata_path = model_path_obj.parent / 'metadata.json'
    if metadata_path.exists():
        import json
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                optimal_threshold = metadata.get('optimal_threshold', 0.5)
                logger.info(
                    f"Loaded optimal threshold: {optimal_threshold:.4f}")
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")

    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Device: {device}")


def predict_text(
    text: str,
    title: Optional[str] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    단일 텍스트 예측

    Args:
        text: 입력 텍스트
        title: 제목 (선택사항)
        threshold: 분류 임계값 (None이면 최적값 사용)

    Returns:
        예측 결과 딕셔너리

    Raises:
        HTTPException: 모델이 로드되지 않은 경우
    """
    global optimal_threshold

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please reload model first."
        )

    # title과 text 결합 (A. 입력 스키마 호환성)
    combined_text = combine_title_text(title, text)

    if threshold is None:
        threshold = optimal_threshold

    # 전처리 및 토크나이즈
    encoded = tokenizer.tokenize(
        [combined_text], padding=True, truncation=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        fake_prob = probabilities[0][1].item()  # Fake 클래스 확률

        # 임계값 기반 분류 (D. 임계값 튜닝)
        predicted_class = 1 if fake_prob >= threshold else 0
        confidence = fake_prob if predicted_class == 1 else probabilities[0][0].item(
        )

    result = {
        'prediction': 'real' if predicted_class == 0 else 'fake',
        'confidence': float(confidence),
        'probabilities': {
            'fake': float(probabilities[0][1].item()),
            'real': float(probabilities[0][0].item())
        },
        'threshold': float(threshold)
    }

    return result


@app.post("/infer", response_model=TextResponse)
async def infer(
    request: TextRequest,
    api_key: str = Depends(verify_api_key)
) -> TextResponse:
    """
    단일 텍스트 추론 (A. title+text 지원)

    - **text**: 판별할 뉴스 본문 텍스트 (필수)
    - **title**: 뉴스 제목 (선택사항, 있으면 자동으로 결합하여 사용)
    """
    try:
        result = predict_text(request.text, request.title)
        return TextResponse(**result)
    except Exception as e:
        logger.error(f"Error in /infer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def read_csv_robust(file_content: bytes) -> pd.DataFrame:
    """
    CSV 파일을 견고하게 읽기 (F. CSV 처리 견고화)
    구분자 자동 감지, 따옴표 처리 등

    Args:
        file_content: 파일 바이트 내용

    Returns:
        DataFrame
    """
    # 여러 방법 시도
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'latin-1']
    separators = [',', ';', '\t']

    for encoding in encodings:
        for sep in separators:
            try:
                file_str = file_content.decode(encoding)
                # csv.Sniffer로 구분자 감지 시도
                sniffer = csv.Sniffer()
                sample = file_str[:1024]
                detected_delimiter = sniffer.sniff(sample).delimiter

                df = pd.read_csv(
                    io.StringIO(file_str),
                    sep=detected_delimiter,
                    encoding=encoding,
                    engine='python',
                    on_bad_lines='skip',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL
                )
                if not df.empty:
                    return df
            except Exception as e:
                logger.debug(
                    f"Failed with encoding={encoding}, sep={sep}: {e}")
                continue

    # 최종 시도
    try:
        return pd.read_csv(io.BytesIO(file_content), encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {str(e)}"
        )


@app.post("/infer_csv", response_model=CSVResponse)
async def infer_csv(
    file: UploadFile = File(...),
    only_prediction: bool = Query(False, description="id,prediction만 반환"),
    api_key: str = Depends(verify_api_key)
) -> CSVResponse:
    """
    CSV 파일 일괄 추론 (F. CSV 처리 견고화)

    CSV 파일은 'text' 컬럼을 포함해야 합니다.
    'title' 컬럼이 있으면 자동으로 함께 사용합니다.

    - **only_prediction**: true이면 id,prediction만 반환
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please reload model first."
        )

    try:
        # CSV 파일 읽기 (견고한 파싱)
        file_content = await file.read()
        df = read_csv_robust(file_content)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty or could not be parsed"
            )

        # 컬럼 자동 탐지 (F. CSV 처리 견고화)
        has_title = 'title' in df.columns
        has_text = 'text' in df.columns

        if not has_text:
            raise HTTPException(
                status_code=400,
                detail="CSV file must contain 'text' column. Available columns: " +
                       ", ".join(df.columns.tolist())
            )

        # title과 text 결합 처리
        if has_title:
            texts = [
                combine_title_text(
                    title if pd.notna(title) else None,
                    text
                )
                for title, text in zip(df['title'], df['text'])
            ]
        else:
            texts = df['text'].astype(str).tolist()

        predictions: List[Dict[str, Any]] = []

        # 청크 처리로 메모리 보호 (F. CSV 처리 견고화)
        chunk_size = 10000  # 10k 라인 단위
        batch_size = 32

        for chunk_start in range(0, len(texts), chunk_size):
            chunk_texts = texts[chunk_start:chunk_start + chunk_size]

            # 배치 처리로 예측
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]

                # 토크나이즈
                encoded = tokenizer.tokenize(
                    batch_texts, padding=True, truncation=True
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                # 예측
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    probabilities = torch.softmax(outputs, dim=1)
                    fake_probs = probabilities[:, 1].cpu().numpy()

                    # 임계값 기반 분류
                    predicted_classes = (
                        fake_probs >= optimal_threshold).astype(int)
                    confidences = np.where(
                        predicted_classes == 1,
                        fake_probs,
                        1 - fake_probs
                    )

                # 결과 저장
                for j, (pred_class, conf) in enumerate(
                    zip(predicted_classes, confidences)
                ):
                    idx = chunk_start + i + j
                    result = {
                        'prediction': 'real' if pred_class == 0 else 'fake',
                        'confidence': float(conf)
                    }

                    if not only_prediction:
                        result['text'] = batch_texts[j]

                    # id 컬럼이 있으면 포함
                    if 'id' in df.columns:
                        result['id'] = str(df.iloc[idx]['id'])

                    predictions.append(result)

        return CSVResponse(predictions=predictions, total=len(predictions))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer_csv: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
async def reload_model(
    model_path: str = "models/best.pt",
    model_type: str = "bilstm",
    api_key: str = Depends(verify_api_key)
) -> Dict[str, str]:
    """
    모델 다시 불러오기

    - **model_path**: 모델 파일 경로 (기본값: models/best.pt)
    - **model_type**: 모델 타입 ('bilstm' or 'cnn')
    """
    try:
        config = load_config(model_name=model_type)
        load_model(model_path, model_type, config)
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "optimal_threshold": optimal_threshold
        }
    except Exception as e:
        logger.error(f"Error in /reload_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate(
    val_ratio: float = 0.2,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    검증 데이터 평가 (B. 라벨 매핑 검증, D. 임계값 튜닝)

    - **val_ratio**: 검증 데이터 비율 (기본값: 0.2)

    반환값에 라벨 매핑 정보와 최적 임계값 포함
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please reload model first."
        )

    try:
        # 데이터 로드
        df = load_datasets(data_dir='data')

        if 'label' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Dataset does not contain 'label' column"
            )

        # 학습/검증 분할
        train_df, val_df = split_train_val(
            df, val_ratio=val_ratio, random_state=42
        )

        # 검증 데이터 준비
        val_texts, val_labels = prepare_data_for_training(val_df)

        if val_labels is None:
            raise HTTPException(
                status_code=400,
                detail="Validation labels are missing"
            )

        # B. 라벨 매핑 검증
        unique_labels = sorted(set(val_labels))
        label_mapping_info = {
            "0": "Real (negative)",
            "1": "Fake (positive)",
            "unique_labels_in_data": unique_labels,
            "label_distribution": {
                str(label): int(sum(1 for l in val_labels if l == label))
                for label in unique_labels
            }
        }

        # 예측
        all_preds: list = []
        all_probs: list = []
        batch_size = 32

        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i:i + batch_size]

            encoded = tokenizer.tokenize(
                batch_texts, padding=True, truncation=True
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                probabilities = torch.softmax(outputs, dim=1)
                fake_probs = probabilities[:, 1].cpu().numpy()
                all_probs.extend(fake_probs)

                # 현재 임계값으로 예측
                preds = (fake_probs >= optimal_threshold).astype(int)
                all_preds.extend(preds)

        # 메트릭 계산
        val_labels_arr = np.array(val_labels)
        metrics = calculate_metrics(val_labels_arr, np.array(all_preds))

        # D. 최적 임계값 탐색
        optimal_threshold_new, optimal_metrics = find_optimal_threshold(
            val_labels_arr, np.array(all_probs)
        )

        # 최적 임계값 업데이트
        global optimal_threshold
        optimal_threshold = optimal_threshold_new

        return {
            "status": "success",
            "metrics": metrics,
            "optimal_threshold": float(optimal_threshold),
            "optimal_threshold_metrics": optimal_metrics,
            "current_threshold_metrics": metrics,
            "label_mapping": label_mapping_info,  # B. 라벨 매핑 정보
            "num_samples": len(val_df)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /validate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> Dict[str, Any]:
    """헬스 체크 (API Key 불필요)"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "optimal_threshold": optimal_threshold
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """루트 엔드포인트 (API Key 불필요)"""
    return {
        "message": "Fake News Detection API",
        "endpoints": [
            "POST /infer - 단일 텍스트 추론 (title+text 지원)",
            "POST /infer_csv - CSV 파일 일괄 추론",
            "POST /reload_model - 모델 다시 불러오기",
            "POST /validate - 검증 데이터 평가",
            "GET /health - 헬스 체크"
        ],
        "note": "All POST endpoints require X-API-Key header",
        "label_mapping": {
            "0": "Real (negative)",
            "1": "Fake (positive)"
        }
    }


@app.on_event("startup")
async def startup_event() -> None:
    """서버 시작 시 실행"""
    model_path = Path("models/best.pt")
    if model_path.exists():
        try:
            config = load_config(model_name='bilstm')
            load_model(str(model_path), 'bilstm', config)
            logger.info("Model loaded on startup")
        except Exception as e:
            logger.warning(f"Failed to load model on startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
