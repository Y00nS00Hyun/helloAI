"""
텍스트 전처리 및 토크나이제이션 유틸리티 (사전학습 모델 없이)
"""
from typing import List, Dict
import torch
import re
from collections import Counter


class SimpleTokenizer:
    """간단한 단어 기반 토크나이저 (사전학습 모델 없음)"""

    def __init__(self, vocab_size: int = 30000, max_length: int = 512):
        """
        Args:
            vocab_size: 어휘 크기
            max_length: 최대 시퀀스 길이
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = None
        self.pad_token_id = 0
        self.unk_token_id = 1

        # 특수 토큰
        self.word_to_idx['<PAD>'] = 0
        self.word_to_idx['<UNK>'] = 1
        self.idx_to_word[0] = '<PAD>'
        self.idx_to_word[1] = '<UNK>'

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        텍스트 리스트로부터 어휘 생성

        Args:
            texts: 텍스트 리스트
            min_freq: 최소 단어 빈도
        """
        # 단어 분리 및 정제
        words = []
        for text in texts:
            text = self._preprocess_text(text)
            words.extend(text.split())

        # 단어 빈도 계산
        word_freq = Counter(words)

        # 빈도 기준으로 정렬
        sorted_words = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True)

        # 어휘 구축 (최소 빈도 이상인 단어만)
        idx = len(self.word_to_idx)  # <PAD>, <UNK> 다음부터
        for word, freq in sorted_words:
            if freq >= min_freq and idx < self.vocab_size:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    idx += 1

        self.vocab = set(self.word_to_idx.keys())
        print(f"Vocabulary size: {len(self.word_to_idx)}")

    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            text = str(text)

        # 소문자 변환
        text = text.lower()

        # 특수문자 제거 (알파벳, 숫자, 공백만 남김)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def encode(self, texts: List[str], padding: bool = True, truncation: bool = True) -> Dict:
        """
        텍스트 리스트를 토큰 ID로 변환

        Args:
            texts: 텍스트 리스트
            padding: 패딩 여부
            truncation: 자르기 여부

        Returns:
            {'input_ids': tensor, 'attention_mask': tensor}
        """
        input_ids_list = []
        attention_mask_list = []

        for text in texts:
            # 전처리
            text = self._preprocess_text(text)
            words = text.split()

            # 단어를 인덱스로 변환
            token_ids = []
            for word in words:
                if word in self.word_to_idx:
                    token_ids.append(self.word_to_idx[word])
                else:
                    token_ids.append(self.unk_token_id)

            # 자르기
            if truncation and len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]

            input_ids_list.append(token_ids)

            # attention mask 생성 (실제 토큰 위치는 1)
            mask = [1] * len(token_ids)
            attention_mask_list.append(mask)

        # 패딩
        if padding:
            max_len = max(len(ids) for ids in input_ids_list)
            if max_len > self.max_length:
                max_len = self.max_length

            padded_ids = []
            padded_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_length = max_len - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_length)
                padded_mask.append(mask + [0] * pad_length)

            input_ids_list = padded_ids
            attention_mask_list = padded_mask

        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
        }


class TextTokenizer:
    """텍스트 토크나이제이션 클래스 (SimpleTokenizer 래퍼)"""

    def __init__(self, vocab_size: int = 30000, max_length: int = 512, model_name: str = None):
        """
        Args:
            vocab_size: 어휘 크기
            max_length: 최대 시퀀스 길이
            model_name: (호환성을 위해 유지, 사용하지 않음)
        """
        self.tokenizer = SimpleTokenizer(
            vocab_size=vocab_size, max_length=max_length)
        self.vocab_built = False

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """어휘 구축"""
        self.tokenizer.build_vocab(texts, min_freq=min_freq)
        self.vocab_built = True

    def tokenize(self, texts: List[str], padding: bool = True, truncation: bool = True) -> Dict:
        """
        텍스트 리스트를 토크나이즈

        Args:
            texts: 텍스트 리스트
            padding: 패딩 여부
            truncation: 자르기 여부

        Returns:
            토크나이즈된 결과 딕셔너리
        """
        if not self.vocab_built:
            # 어휘가 구축되지 않은 경우 즉시 구축
            self.build_vocab(texts)

        return self.tokenizer.encode(texts, padding=padding, truncation=truncation)

    def tokenize_batch(self, texts: List[str], batch_size: int = 32) -> Dict:
        """
        배치 단위로 토크나이즈 (간단한 경우는 그냥 전체 처리)

        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기

        Returns:
            토크나이즈된 결과 딕셔너리
        """
        return self.tokenize(texts)
