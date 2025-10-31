"""
데이터 분할 스크립트
원본 dataset_1.csv, dataset_2.csv, dataset_3.csv를 train.csv와 validation.csv로 분할
"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_combine_datasets(data_dir: str = "data"):
    """
    3개의 데이터셋을 로드하고 통합
    
    Args:
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        통합된 DataFrame
    """
    data_dir = Path(data_dir)
    dfs = []
    
    # dataset_1.csv 및 dataset_2.csv (title, text 포함)
    for i in [1, 2]:
        file_path = data_dir / f"dataset_{i}.csv"
        if file_path.exists():
            try:
                # 다양한 구분자 시도
                df = pd.read_csv(file_path, encoding='utf-8', sep=',', on_bad_lines='skip')
            except:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', sep=';', on_bad_lines='skip')
                except:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            
            # title과 text를 결합
            if 'title' in df.columns and 'text' in df.columns:
                df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
            elif 'title' in df.columns:
                df['text'] = df['title'].astype(str)
            
            # label 컬럼 확인
            if 'label' in df.columns:
                dfs.append(df[['text', 'label']])
            else:
                dfs.append(df[['text']])
    
    # dataset_3.csv (text만 존재)
    file_path = data_dir / "dataset_3.csv"
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=',', on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', sep=';', on_bad_lines='skip')
            except:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        if 'text' in df.columns:
            if 'label' in df.columns:
                dfs.append(df[['text', 'label']])
            else:
                dfs.append(df[['text']])
    
    # 모든 데이터프레임 결합
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 중복 제거
        if 'label' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        
        return combined_df
    else:
        return pd.DataFrame()


def normalize_labels(df: pd.DataFrame):
    """
    라벨을 정수로 정규화 (Real=0, Fake=1)
    
    Args:
        df: 데이터프레임
    
    Returns:
        정규화된 데이터프레임
    """
    if 'label' not in df.columns:
        return df
    
    df = df.copy()
    
    # 라벨을 정수로 변환
    label_mapping = {
        'real': 0, 'Real': 0, 'REAL': 0,
        'fake': 1, 'Fake': 1, 'FAKE': 1,
        0: 0, 1: 1,
        '0': 0, '1': 1,
        True: 1, False: 0
    }
    
    def map_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).strip()
        if label_str in label_mapping:
            return label_mapping[label_str]
        try:
            label_int = int(float(label_str))
            return 0 if label_int == 0 else 1
        except:
            return 0
    
    df['label'] = df['label'].apply(map_label)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train and validation')
    parser.add_argument('--ratio', type=float, default=0.8,
                       help='Train ratio (default: 0.8)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for splitting (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Loading datasets from {args.data_dir}...")
    df = load_and_combine_datasets(data_dir=args.data_dir)
    
    if df.empty:
        print("Error: No data loaded!")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # 라벨 정규화
    if 'label' in df.columns:
        df = normalize_labels(df)
        print(f"Label distribution:")
        print(df['label'].value_counts().sort_index())
        print("(0=Real, 1=Fake)")
        
        # 학습/검증 분할 (stratified)
        train_df, val_df = train_test_split(
            df,
            test_size=1 - args.ratio,
            random_state=args.random_state,
            stratify=df['label']
        )
    else:
        print("Warning: No label column found. Splitting without stratification.")
        train_df, val_df = train_test_split(
            df,
            test_size=1 - args.ratio,
            random_state=args.random_state
        )
    
    print(f"\nSplit results:")
    print(f"Train: {len(train_df)} samples ({args.ratio*100:.1f}%)")
    print(f"Validation: {len(val_df)} samples ({(1-args.ratio)*100:.1f}%)")
    
    # 저장
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    train_path = data_dir / 'train.csv'
    val_path = data_dir / 'validation.csv'
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    
    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")


if __name__ == '__main__':
    main()

