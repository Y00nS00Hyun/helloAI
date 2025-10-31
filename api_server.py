"""
FastAPI 서버 - 가짜뉴스 판별 API
"""
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from utils.data_loader import (
    load_datasets,
    split_train_val,
    prepare_data_for_training
)
from utils.preprocessing import TextTokenizer
from utils.metrics import calculate_metrics
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


# Pydantic 모델
class TextRequest(BaseModel):
    """단일 텍스트 추론 요청"""
    text: str


class TextResponse(BaseModel):
    """단일 텍스트 추론 응답"""
    prediction: str  # 'real' or 'fake'
    confidence: float
    probabilities: Dict[str, float]


class CSVResponse(BaseModel):
    """CSV 추론 응답"""
    predictions: List[Dict[str, Any]]
    total: int


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
    global model, tokenizer, model_config, model_type

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

    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Device: {device}")


def predict_text(text: str) -> Dict[str, Any]:
    """
    단일 텍스트 예측

    Args:
        text: 입력 텍스트

    Returns:
        예측 결과 딕셔너리

    Raises:
        HTTPException: 모델이 로드되지 않은 경우
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please reload model first."
        )

    # 전처리 및 토크나이즈
    encoded = tokenizer.tokenize([text], padding=True, truncation=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    result = {
        'prediction': 'real' if predicted_class == 0 else 'fake',
        'confidence': float(confidence),
        'probabilities': {
            'fake': float(probabilities[0][1].item()),
            'real': float(probabilities[0][0].item())
        }
    }

    return result


@app.post("/infer", response_model=TextResponse)
async def infer(
    request: TextRequest,
    api_key: str = Depends(verify_api_key)
) -> TextResponse:
    """
    단일 텍스트 추론

    - **text**: 판별할 뉴스 텍스트
    """
    try:
        result = predict_text(request.text)
        return TextResponse(**result)
    except Exception as e:
        logger.error(f"Error in /infer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_csv", response_model=CSVResponse)
async def infer_csv(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
) -> CSVResponse:
    """
    CSV 파일 일괄 추론

    CSV 파일은 'text' 컬럼을 포함해야 합니다.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please reload model first."
        )

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file.file, encoding='utf-8')

        if 'text' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV file must contain 'text' column"
            )

        texts = df['text'].astype(str).tolist()
        predictions: List[Dict[str, Any]] = []

        # 배치 처리로 예측
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

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
                predicted_classes = torch.argmax(
                    probabilities, dim=1
                ).cpu().numpy()
                confidences = torch.max(
                    probabilities, dim=1
                )[0].cpu().numpy()

            # 결과 저장
            for j, (pred_class, conf) in enumerate(
                zip(predicted_classes, confidences)
            ):
                predictions.append({
                    'text': batch_texts[j],
                    'prediction': 'real' if pred_class == 0 else 'fake',
                    'confidence': float(conf)
                })

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
    - **model_type**: 모델 타입 ('bilstm' or 'transformer')
    """
    try:
        config = load_config(model_name=model_type)
        load_model(model_path, model_type, config)
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}"
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
    검증 데이터 평가

    - **val_ratio**: 검증 데이터 비율 (기본값: 0.2)
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

        # 예측
        all_preds: list = []
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
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

        # 메트릭 계산
        metrics = calculate_metrics(
            np.array(val_labels),
            np.array(all_preds)
        )

        return {
            "status": "success",
            "metrics": metrics,
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
        "device": str(device)
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """루트 엔드포인트 (API Key 불필요)"""
    return {
        "message": "Fake News Detection API",
        "endpoints": [
            "POST /infer - 단일 텍스트 추론",
            "POST /infer_csv - CSV 파일 일괄 추론",
            "POST /reload_model - 모델 다시 불러오기",
            "POST /validate - 검증 데이터 평가",
            "GET /health - 헬스 체크"
        ],
        "note": "All POST endpoints require X-API-Key header"
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
