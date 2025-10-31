"""
모델 학습 스크립트
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from utils.data_loader import load_datasets, split_train_val, prepare_data_for_training
from utils.preprocessing import TextTokenizer
from utils.metrics import calculate_metrics, print_classification_report
from model_definitions import BiLSTMModel, TransformerModel


class NewsDataset(Dataset):
    """뉴스 데이터셋"""
    
    def __init__(self, texts: list, labels: list = None, tokenizer: TextTokenizer = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer:
            encoded = self.tokenizer.tokenize([text], padding=True, truncation=True)
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            # 기본 토크나이제이션 (간단한 경우)
            input_ids = torch.tensor([hash(word) % 30000 for word in text.split()[:512]], dtype=torch.long)
            attention_mask = torch.ones(len(input_ids))
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


def get_scheduler(optimizer, config):
    """스케줄러 생성"""
    scheduler_type = config.get('scheduler')
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    elif scheduler_type == 'linear_warmup':
        # 간단한 linear warmup 구현 (transformers 라이브러리 없이)
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step):
            warmup_steps = int(0.1 * config['num_epochs'] * 1000)
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        return LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Train Fake News Detection Model')
    parser.add_argument('--model', type=str, required=True, choices=['bilstm', 'transformer'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config_path = f'configs/{args.model}.yaml'
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 기본 설정
            config = {
                'model': {'name': args.model},
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'num_epochs': 10,
                    'val_ratio': 0.2
                },
                'data': {'data_dir': 'data'},
                'tokenizer': {'vocab_size': 30000, 'max_length': 512}
            }
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 데이터 로드
    print("Loading datasets...")
    df = load_datasets(data_dir=config['data'].get('data_dir', 'data'))
    print(f"Loaded {len(df)} samples")
    
    # 학습/검증 분할
    train_df, val_df = split_train_val(
        df,
        val_ratio=config['training'].get('val_ratio', 0.2),
        random_state=config['training'].get('random_state', 42)
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 데이터 준비
    train_texts, train_labels = prepare_data_for_training(train_df)
    val_texts, val_labels = prepare_data_for_training(val_df)
    
    # 토크나이저 설정 (사전학습 모델 없이)
    tokenizer_config = config.get('tokenizer', {})
    tokenizer = TextTokenizer(
        vocab_size=tokenizer_config.get('vocab_size', 30000),
        max_length=tokenizer_config.get('max_length', 512)
    )
    
    # 어휘 구축 (학습 데이터 사용)
    print("Building vocabulary...")
    tokenizer.build_vocab(train_texts, min_freq=2)
    print(f"Vocabulary size: {len(tokenizer.tokenizer.word_to_idx)}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=False,
        num_workers=0
    )
    
    # 모델 생성 (사전학습 모델 없이)
    model_config = config['model']
    vocab_size = len(tokenizer.tokenizer.word_to_idx)
    
    if args.model == 'bilstm':
        model = BiLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=model_config.get('embedding_dim', 300),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    elif args.model == 'transformer':
        # CNN 모델 사용 (Transformer 대신)
        model = TransformerModel(
            model_name=None,
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3),
            vocab_size=vocab_size
        )
    
    model = model.to(device)
    print(f"Model: {args.model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    
    train_config = config['training']
    lr = train_config.get('learning_rate', 0.001)
    
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=train_config.get('weight_decay', 0))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_config.get('weight_decay', 0))
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=train_config.get('weight_decay', 0))
    
    scheduler = get_scheduler(optimizer, train_config)
    
    # 학습 루프
    num_epochs = train_config.get('num_epochs', 10)
    best_f1 = 0
    patience_counter = 0
    early_stopping_config = train_config.get('early_stopping', {})
    patience = early_stopping_config.get('patience', 5)
    min_delta = early_stopping_config.get('min_delta', 0.001)
    
    # 모델 저장 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 학습
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1_score']:.4f}")
        
        # 검증
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1_score']:.4f}")
        
        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 최고 모델 저장
        if val_metrics['f1_score'] > best_f1 + min_delta:
            best_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), 'models/best.pt')
            print(f"✓ Best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*50)
    print("Training Completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print("="*50)
    
    # 최종 검증 리포트
    model.load_state_dict(torch.load('models/best.pt'))
    val_loss, val_metrics = validate(model, val_loader, criterion, device)
    print("\nFinal Validation Results:")
    print(f"F1 Score: {val_metrics['f1_score']:.4f}")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall: {val_metrics['recall']:.4f}")


if __name__ == '__main__':
    main()

