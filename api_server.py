"""
FastAPI 서버 - 가짜뉴스 판별 API
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import os

from utils.data_loader import load_datasets, split_train_val, prepare_data_for_training
from utils.preprocessing import TextTokenizer
from utils.metrics import calculate_metrics
from model_definitions import MODEL_REGISTRY
from train import NewsDataset

app = FastAPI(title="Fake News Detection API", version="1.0.0")

# API Key 인증 설정 (환경변수 또는 설정에서 로드)
API_KEY = os.getenv("API_KEY", "your-api-key-here")  # 실제로는 환경변수나 설정 파일에서 로드
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """API Key 검증"""
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key. Please provide valid X-API-Key header."
        )
    return api_key


# 전역 변수
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config = None
model_type = None


class TextRequest(BaseModel):
    """단일 텍스트 추론 요청"""
    text: str


class TextResponse(BaseModel):
    """단일 텍스트 추론 응답"""
    prediction: str  # 'real' or 'fake'
    confidence: float
    probabilities: dict


class CSVResponse(BaseModel):
    """CSV 추론 응답"""
    predictions: List[dict]
    total: int


def load_model(model_path: str = "models/best.pt", model_type_param: str = None, config: dict = None):
    """모델 로드"""
    global model, tokenizer, model_config, model_type

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 모델 타입 및 설정 확인 (기본값 사용)
    if not model_type_param:
        model_type_param = "bilstm"  # 기본값을 bilstm으로 변경 (사전학습 없음)
        if config is None:
            config = {
                'model': {
                    'name': 'bilstm',
                    'vocab_size': 30000,
                    'embedding_dim': 300,
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'num_classes': 2,
                    'dropout': 0.3
                },
                'tokenizer': {
                    'vocab_size': 30000,
                    'max_length': 512
                }
            }

    model_config = config
    model_type = model_type_param

    # 토크나이저 로드 (사전학습 모델 없이)
    tokenizer_config = config.get('tokenizer', {})
    tokenizer = TextTokenizer(
        vocab_size=tokenizer_config.get('vocab_size', 30000),
        max_length=tokenizer_config.get('max_length', 512)
    )

    # 학습 데이터로 어휘 구축 (모델 로드 시 필요)
    try:
        df = load_datasets(data_dir='data')
        if 'label' in df.columns:
            train_df, _ = split_train_val(df, val_ratio=0.2, random_state=42)
            train_texts, _ = prepare_data_for_training(train_df)
            tokenizer.build_vocab(train_texts, min_freq=2)
            print(
                f"Vocabulary built with {len(tokenizer.tokenizer.word_to_idx)} tokens")
        else:
            # 라벨이 없으면 전체 데이터로 어휘 구축
            texts, _ = prepare_data_for_training(df)
            tokenizer.build_vocab(texts, min_freq=2)
            print(
                f"Vocabulary built with {len(tokenizer.tokenizer.word_to_idx)} tokens")
    except Exception as e:
        print(f"Warning: Could not build vocabulary: {e}")
        # 기본 어휘 사용

    # 모델 생성 (레지스트리에서 가져오기)
    if model_type_param not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type_param}. Available: {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[model_type_param]
    model_config_dict = config.get('model', {})
    vocab_size = len(tokenizer.tokenizer.word_to_idx) if hasattr(tokenizer.tokenizer,
                                                                 'word_to_idx') and tokenizer.tokenizer.word_to_idx else model_config_dict.get('vocab_size', 30000)

    if model_type_param == 'bilstm':
        model = ModelClass(
            vocab_size=vocab_size,
            embedding_dim=model_config_dict.get('embedding_dim', 300),
            hidden_dim=model_config_dict.get('hidden_dim', 256),
            num_layers=model_config_dict.get('num_layers', 2),
            num_classes=model_config_dict.get('num_classes', 2),
            dropout=model_config_dict.get('dropout', 0.3)
        )
    elif model_type_param == 'transformer' or model_type_param == 'cnn':
        # CNN 모델 사용
        model = ModelClass(
            model_name=None,
            num_classes=model_config_dict.get('num_classes', 2),
            dropout=model_config_dict.get('dropout', 0.3),
            vocab_size=vocab_size
        )
    else:
        # 다른 모델의 경우 기본 파라미터로 생성 시도
        try:
            model = ModelClass(
                vocab_size=vocab_size,
                num_classes=model_config_dict.get('num_classes', 2),
                dropout=model_config_dict.get('dropout', 0.3),
                **{k: v for k, v in model_config_dict.items() if k not in ['name', 'model_name']}
            )
        except TypeError as e:
            raise ValueError(
                f"Model {model_type_param} requires specific parameters. Error: {e}")

    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")


def predict_text(text: str) -> dict:
    """단일 텍스트 예측"""
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please reload model first.")

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
        'prediction': 'real' if predicted_class == 1 else 'fake',
        'confidence': float(confidence),
        'probabilities': {
            'fake': float(probabilities[0][0].item()),
            'real': float(probabilities[0][1].item())
        }
    }

    return result


@app.post("/infer", response_model=TextResponse)
async def infer(
    request: TextRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    단일 텍스트 추론

    - **text**: 판별할 뉴스 텍스트
    """
    try:
        result = predict_text(request.text)
        return TextResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_csv", response_model=CSVResponse)
async def infer_csv(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    CSV 파일 일괄 추론

    CSV 파일은 'text' 컬럼을 포함해야 합니다.
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please reload model first.")

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file.file, encoding='utf-8')

        if 'text' not in df.columns:
            raise HTTPException(
                status_code=400, detail="CSV file must contain 'text' column")

        texts = df['text'].astype(str).tolist()
        predictions = []

        # 배치 처리로 예측
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 토크나이즈
            encoded = tokenizer.tokenize(
                batch_texts, padding=True, truncation=True)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # 예측
            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(
                    probabilities, dim=1).cpu().numpy()
                confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()

            # 결과 저장
            for j, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
                predictions.append({
                    'text': batch_texts[j],
                    'prediction': 'real' if pred_class == 1 else 'fake',
                    'confidence': float(conf)
                })

        return CSVResponse(predictions=predictions, total=len(predictions))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
async def reload_model(
    model_path: str = "models/best.pt",
    model_type: str = "bilstm",
    api_key: str = Depends(verify_api_key)
):
    """
    모델 다시 불러오기

    - **model_path**: 모델 파일 경로 (기본값: models/best.pt)
    - **model_type**: 모델 타입 ('bilstm' or 'transformer')
    """
    try:
        # 설정 파일에서 읽기 시도
        import yaml
        config_path = f'configs/{model_type}.yaml'
        config = None

        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 기본 설정 (사전학습 모델 없이)
            config = {
                'model': {
                    'name': model_type,
                    'vocab_size': 30000,
                    'embedding_dim': 300,
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'num_classes': 2,
                    'dropout': 0.3
                },
                'tokenizer': {
                    'vocab_size': 30000,
                    'max_length': 512
                }
            }

        load_model(model_path, model_type, config)
        return {"status": "success", "message": f"Model loaded from {model_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate(
    val_ratio: float = 0.2,
    api_key: str = Depends(verify_api_key)
):
    """
    검증 데이터 평가

    - **val_ratio**: 검증 데이터 비율 (기본값: 0.2)
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please reload model first.")

    try:
        # 데이터 로드
        df = load_datasets(data_dir='data')

        if 'label' not in df.columns:
            raise HTTPException(
                status_code=400, detail="Dataset does not contain 'label' column")

        # 학습/검증 분할
        train_df, val_df = split_train_val(
            df, val_ratio=val_ratio, random_state=42)

        # 검증 데이터 준비
        val_texts, val_labels = prepare_data_for_training(val_df)

        # 예측
        all_preds = []

        batch_size = 32
        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i:i + batch_size]

            encoded = tokenizer.tokenize(
                batch_texts, padding=True, truncation=True)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

        # 메트릭 계산
        metrics = calculate_metrics(np.array(val_labels), np.array(all_preds))

        return {
            "status": "success",
            "metrics": metrics,
            "num_samples": len(val_df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """헬스 체크 (API Key 불필요)"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/")
async def root():
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


# 서버 시작 시 모델 로드 시도
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    model_path = "models/best.pt"
    if os.path.exists(model_path):
        try:
            # 기본 설정으로 모델 로드 시도
            config = {
                'model': {
                    'name': 'bilstm',
                    'vocab_size': 30000,
                    'embedding_dim': 300,
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'num_classes': 2,
                    'dropout': 0.3
                },
                'tokenizer': {
                    'vocab_size': 30000,
                    'max_length': 512
                }
            }
            load_model(model_path, model_type='bilstm', config=config)
            print("Model loaded on startup")
        except Exception as e:
            print(f"Failed to load model on startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
