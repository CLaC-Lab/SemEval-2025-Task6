import json
import logging
import math
import os
import random
import torch
import warnings
import gc
from typing import Dict, List, Tuple

if torch.backends.mps.is_available():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def predict_with_tta(
        model: nn.Module,
        texts: List[str],
        tokenizer,
        device: torch.device,
        n_tta: int = 5,
        batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced test time augmentation with multiple strategies"""
    model.eval()
    all_promise_probs = []
    all_evidence_probs = []

    augmentation_strategies = [
        lambda x: x,  # Original
        lambda x: ' '.join(x.split()),  # Basic normalization
        lambda x: ' '.join([w for w in x.split() if random.random() > 0.1]),  # Word dropout
        lambda x: '[ESG REPORT] ' + x,  # Add ESG context
        lambda x: ' '.join(x.split()[::-1])  # Reverse word order
    ]

    for tta_idx in tqdm(range(n_tta), desc="TTA passes"):
        promise_probs = []
        evidence_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing batch (TTA {tta_idx + 1}/{n_tta})"):
            batch_texts = texts[i:i + batch_size]

            # Apply augmentations
            if tta_idx > 0:
                aug_strategy = random.choice(augmentation_strategies)
                batch_texts = [aug_strategy(text) for text in batch_texts]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
                return_token_type_ids=True
            ).to(device)

            with torch.no_grad():
                promise_logits, evidence_logits = model(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    token_type_ids=encodings['token_type_ids']
                )

                # Get probabilities with temperature scaling
                temperature = 1.5
                promise_prob = torch.softmax(promise_logits / temperature, dim=1)[:, 1].cpu().numpy()
                evidence_prob = torch.softmax(evidence_logits / temperature, dim=1)[:, 1].cpu().numpy()

                promise_probs.extend(promise_prob)
                evidence_probs.extend(evidence_prob)

            try_empty_cache()

        all_promise_probs.append(promise_probs)
        all_evidence_probs.append(evidence_probs)

    # Weighted averaging of predictions
    weights = np.linspace(1.0, 0.6, n_tta)  # Give more weight to earlier passes
    weights = weights / weights.sum()

    final_promise_probs = np.average(all_promise_probs, axis=0, weights=weights)
    final_evidence_probs = np.average(all_evidence_probs, axis=0, weights=weights)

    return final_promise_probs, final_evidence_probs

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        logger.info("Using MPS device")
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
    elif torch.cuda.is_available():
        logger.info("Using CUDA device")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU device")
        device = torch.device("cpu")
    return device


def try_empty_cache():
    """Try to empty cache for the appropriate device"""
    try:
        if torch.backends.mps.is_available():
            try:
                import torch.mps
                torch.mps.empty_cache()
                # Additional MPS memory cleanup
                gc.collect()
            except:
                pass
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except:
        pass


class ESGDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            page_nums: List[int],
            urls: List[str],
            promise_labels: List[str],
            evidence_labels: List[str],
            tokenizer,
            max_length: int = 128,  # Reduced from 256
            augment: bool = False
    ):
        self.texts = texts
        self.page_nums = page_nums
        self.urls = urls
        self.promise_labels = promise_labels
        self.evidence_labels = evidence_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def _augment_text(self, text: str) -> str:
        """Simple text augmentation strategies"""
        if random.random() < 0.5:
            # Random word dropout
            words = text.split()
            if len(words) > 10:  # Only drop if we have enough words
                dropout_idx = random.sample(range(len(words)), k=int(len(words) * 0.1))
                words = [w for i, w in enumerate(words) if i not in dropout_idx]
                text = ' '.join(words)
        return text

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        if self.augment:
            text = self._augment_text(text)

        page_num = self.page_nums[idx]
        url = self.urls[idx]

        # Enrich context with metadata
        enriched_text = f"[PAGE {page_num}] {text}"
        if any(keyword in url.lower() for keyword in
               ['sustainability', 'esg', 'environmental', 'social', 'governance']):
            enriched_text = "[ESG REPORT] " + enriched_text

        # Tokenize with attention to special tokens
        encoding = self.tokenizer(
            enriched_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'promise_label': torch.tensor(1 if self.promise_labels[idx] == "Yes" else 0, dtype=torch.long),
            'evidence_label': torch.tensor(self.evidence_to_id(self.evidence_labels[idx]), dtype=torch.long)
        }

    @staticmethod
    def evidence_to_id(evidence: str) -> int:
        if evidence == "No" or evidence == "N/A":
            return 0
        elif evidence == "Yes":
            return 1
        return 0


class ESGModel(nn.Module):
    def __init__(
            self,
            model_name: str = "roberta-large",
            dropout: float = 0.3,
            num_labels_promise: int = 2,
            num_labels_evidence: int = 2,
            max_length: int = 512
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout
        self.config.gradient_checkpointing = True
        self.config.use_cache = False

        # Initialize base model
        self.base_model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        hidden_size = self.base_model.config.hidden_size

        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(4)
        ])

        # Promise classifier with residual connections
        self.promise_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.GELU(),
            ) for _ in range(2)
        ])
        self.promise_output = nn.Linear(hidden_size, num_labels_promise)

        # Evidence classifier with residual connections
        self.evidence_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.GELU(),
            ) for _ in range(2)
        ])
        self.evidence_output = nn.Linear(hidden_size, num_labels_evidence)

        # Initialize weights
        self._init_weights()

    def get_dynamic_loss_weights(epoch, max_epochs):
        alpha = 0.5 + 0.1 * math.sin(math.pi * epoch / max_epochs)
        return alpha, 1 - alpha

    def get_layer_wise_lr(model, base_lr=5e-6, decay=0.95):
        lr_dict = {}
        num_layers = model.base_model.config.num_hidden_layers

        # Decay learning rate for deeper layers
        for i in range(num_layers):
            lr_dict[f'layer.{i}.'] = base_lr * (decay ** i)

        # Higher learning rate for classifier heads
        lr_dict['classifier'] = base_lr * 2
        return lr_dict

    def _init_weights(self):
        """Initialize weights with controlled values"""
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weights)

    def attention_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Multi-head attention pooling"""
        attention_mask = attention_mask.unsqueeze(-1)

        # Calculate attention from each head
        attention_outputs = []
        for head in self.attention_heads:
            attention_weights = head(sequence_output)
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
            attention_probs = F.softmax(attention_weights, dim=1)
            attention_output = torch.sum(sequence_output * attention_probs, dim=1)
            attention_outputs.append(attention_output)

        # Combine attention outputs
        pooled_output = torch.mean(torch.stack(attention_outputs), dim=0)
        return pooled_output

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = self.attention_pooling(sequence_output, attention_mask)

        # Promise classification with residual connections
        promise_hidden = pooled_output
        for layer in self.promise_layers:
            promise_hidden = layer(promise_hidden) + promise_hidden
        promise_logits = self.promise_output(promise_hidden)

        # Evidence classification with residual connections
        evidence_hidden = pooled_output
        for layer in self.evidence_layers:
            evidence_hidden = layer(evidence_hidden) + evidence_hidden
        evidence_logits = self.evidence_output(evidence_hidden)

        return promise_logits, evidence_logits


class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-7

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int = None) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Dynamic scaling based on training progress
        if epoch is not None:
            self.alpha = min(0.5 + 0.1 * (epoch / 10), 0.8)
            self.gamma = max(1.5 - 0.1 * (epoch / 10), 1.0)

        focal_loss = self.alpha * (1 - pt + self.epsilon) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_layer_wise_lr(model, base_lr=5e-6, decay=0.95):
    """Get layer-wise learning rates with decay for deeper layers"""
    lr_dict = {}
    num_layers = model.base_model.config.num_hidden_layers

    # Base model layers
    for i in range(num_layers):
        lr_dict[f'base_model.encoder.layer.{i}.'] = base_lr * (decay ** i)

    # Higher learning rate for classifiers
    lr_dict['promise_layers'] = base_lr * 2
    lr_dict['evidence_layers'] = base_lr * 2
    lr_dict['attention_heads'] = base_lr * 1.5

    return lr_dict


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
        device: torch.device = None,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.2,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 0.5
):
    if device is None:
        device = get_device()

    scaler = GradScaler()

    model = model.to(device)
    logger.info(f"Model moved to {device}")

    try:
        # Get layer-wise learning rates
        lr_dict = get_layer_wise_lr(model, base_lr=learning_rate)

        # Initialize optimizer with layer-wise learning rates
        optimizer_grouped_parameters = []
        for name, param in model.named_parameters():
            # Find matching lr from lr_dict
            lr = next((lr for key, lr in lr_dict.items() if key in name), learning_rate)

            if any(nd in name for nd in ['bias', 'LayerNorm.weight']):
                optimizer_grouped_parameters.append({
                    'params': [param],
                    'weight_decay': 0.0,
                    'lr': lr
                })
            else:
                optimizer_grouped_parameters.append({
                    'params': [param],
                    'weight_decay': weight_decay,
                    'lr': lr
                })

        optimizer = AdamW(optimizer_grouped_parameters)

        # Calculate scheduler steps
        num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        max_train_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps = int(max_train_steps * warmup_ratio)

        # Use cosine scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps
        )

        # Initialize loss functions
        promise_criterion = DynamicFocalLoss()
        evidence_criterion = DynamicFocalLoss()

        # Training state tracking
        best_val_loss = float('inf')
        best_f1 = 0.0
        patience = 2
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for step, batch in enumerate(pbar):
                    try_empty_cache()

                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    promise_labels = batch['promise_label'].to(device)
                    evidence_labels = batch['evidence_label'].to(device)

                    # Use automatic mixed precision
                    with autocast():
                        promise_logits, evidence_logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )

                        # Calculate losses with dynamic weighting
                        promise_loss = promise_criterion(promise_logits, promise_labels, epoch)
                        evidence_loss = evidence_criterion(evidence_logits, evidence_labels, epoch)

                        # Dynamic loss weighting
                        alpha = 0.5 + 0.1 * math.sin(math.pi * epoch / num_epochs)
                        loss = alpha * promise_loss + (1 - alpha) * evidence_loss
                        loss = loss / gradient_accumulation_steps

                    # Scale loss and backward pass
                    scaler.scale(loss).backward()

                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Unscale gradients and clip
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        # Step optimizer and scaler
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                    total_loss += loss.item() * gradient_accumulation_steps
                    pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

            # Validation phase
            model.eval()
            val_loss = 0
            promise_preds = []
            promise_labels_val = []
            evidence_preds = []
            evidence_labels_val = []

            with torch.no_grad():
                for batch in val_loader:
                    try_empty_cache()

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    promise_labels = batch['promise_label'].to(device)
                    evidence_labels = batch['evidence_label'].to(device)

                    promise_logits, evidence_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )

                    # Calculate validation loss
                    promise_loss = promise_criterion(promise_logits, promise_labels)
                    evidence_loss = evidence_criterion(evidence_logits, evidence_labels)
                    loss = 0.5 * promise_loss + 0.5 * evidence_loss
                    val_loss += loss.item()

                    # Store predictions for F1 calculation
                    promise_preds.extend(torch.argmax(promise_logits, dim=1).cpu().numpy())
                    promise_labels_val.extend(promise_labels.cpu().numpy())
                    evidence_preds.extend(torch.argmax(evidence_logits, dim=1).cpu().numpy())
                    evidence_labels_val.extend(evidence_labels.cpu().numpy())

            val_loss = val_loss / len(val_loader)
            promise_f1 = f1_score(promise_labels_val, promise_preds, average='macro')
            evidence_f1 = f1_score(evidence_labels_val, evidence_preds, average='macro')
            avg_f1 = (promise_f1 + evidence_f1) / 2

            logger.info(f'Epoch {epoch + 1}, Validation loss: {val_loss:.4f}, Avg F1: {avg_f1:.4f}')
            logger.info(f'Promise F1: {promise_f1:.4f}, Evidence F1: {evidence_f1:.4f}')

            # Save best model and check for early stopping
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                logger.info(f'Saving best model with F1: {avg_f1:.4f}')
                torch.save(model.state_dict(), '../../data/best_model.pt')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping triggered")
                    break

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        try_empty_cache()


def predict_and_save(
        model: nn.Module,
        test_file: str,
        output_file: str,
        device: torch.device = None,
        batch_size: int = 8
):
    """Generate predictions with test time augmentation and ensemble"""
    if device is None:
        device = get_device()

    logger.info(f"Loading test data from {test_file}")
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading test file: {e}")
        return

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    # Extract texts ensuring order is maintained
    texts = []
    for idx in range(len(test_data["ID"])):
        str_idx = str(idx)
        if str_idx in test_data["data"]:
            texts.append(test_data["data"][str_idx].strip())
        else:
            texts.append("")  # Handle missing data gracefully

    logger.info(f"Processing {len(texts)} texts")

    all_promise_probs = []
    all_evidence_probs = []

    # Test Time Augmentation passes
    n_tta = 3  # Number of TTA passes

    for tta_idx in tqdm(range(n_tta), desc="TTA passes"):
        promise_probs = []
        evidence_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing batch (TTA {tta_idx + 1}/{n_tta})"):
            batch_texts = texts[i:i + batch_size]

            # Apply different augmentations for each TTA pass
            if tta_idx > 0:
                batch_texts = [text if random.random() > 0.3 else
                               ' '.join([w for w in text.split() if random.random() > 0.1])
                               for text in batch_texts]

            # Prepare batch with metadata
            enriched_texts = []
            for text in batch_texts:
                if random.random() > 0.5 and tta_idx > 0:
                    enriched_text = f"[ESG REPORT] {text}"
                else:
                    enriched_text = text
                enriched_texts.append(enriched_text)

            # Tokenize
            encodings = tokenizer(
                enriched_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=True
            ).to(device)

            with torch.no_grad():
                promise_logits, evidence_logits = model(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    token_type_ids=encodings['token_type_ids']
                )

                # Get probabilities
                promise_prob = torch.softmax(promise_logits, dim=1)[:, 1].cpu().numpy()
                evidence_prob = torch.softmax(evidence_logits, dim=1)[:, 1].cpu().numpy()

                promise_probs.extend(promise_prob)
                evidence_probs.extend(evidence_prob)

            try_empty_cache()

        all_promise_probs.append(promise_probs)
        all_evidence_probs.append(evidence_probs)

    # Average predictions from all TTA passes
    final_promise_probs = np.mean(all_promise_probs, axis=0)
    final_evidence_probs = np.mean(all_evidence_probs, axis=0)

    # Convert probabilities to predictions
    promise_threshold = 0.5
    evidence_threshold = 0.5

    promise_preds = (final_promise_probs > promise_threshold)
    evidence_preds = (final_evidence_probs > evidence_threshold)

    # Update predictions in data
    for idx in range(len(texts)):
        str_idx = str(idx)
        test_data["promise_status"][str_idx] = "Yes" if promise_preds[idx] else "No"
        test_data["evidence_status"][str_idx] = "Yes" if evidence_preds[idx] else "No"
        test_data["verification_timeline"][str_idx] = "N/A" if not promise_preds[idx] else "Less than 2 years"
        test_data["evidence_quality"][str_idx] = "N/A" if not evidence_preds[idx] else "Clear"

    # Save predictions
    logger.info(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    # Save to parquet
    try:
        logger.info("Converting to DataFrame for parquet export...")
        df = pd.DataFrame({
            'ID': list(test_data['ID'].values()),
            'data': [test_data['data'].get(str(i), "") for i in range(len(test_data['ID']))],
            'URL': [test_data['URL'].get(str(i), "") for i in range(len(test_data['ID']))],
            'page_number': [test_data['page_number'].get(str(i), "N/A") for i in range(len(test_data['ID']))],
            'promise_status': [test_data['promise_status'].get(str(i), "No") for i in range(len(test_data['ID']))],
            'verification_timeline': [test_data['verification_timeline'].get(str(i), "N/A") for i in
                                      range(len(test_data['ID']))],
            'evidence_status': [test_data['evidence_status'].get(str(i), "No") for i in range(len(test_data['ID']))],
            'evidence_quality': [test_data['evidence_quality'].get(str(i), "N/A") for i in range(len(test_data['ID']))]
        })

        parquet_path = output_file.rsplit('.', 1)[0] + '.parquet'
        logger.info(f"Saving parquet file to {parquet_path}")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path, compression='snappy')
        logger.info("Successfully saved parquet file")

    except Exception as e:
        logger.error(f"Failed to save Parquet file: {str(e)}")


def prepare_data(data_file: str) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
    """Prepare data from JSON file with improved preprocessing"""
    logger.info(f"Loading data from {data_file}")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        page_nums = []
        urls = []
        promise_labels = []
        evidence_labels = []

        num_samples = len(data["ID"]) if isinstance(data, dict) else len(data)
        for idx in range(num_samples):
            str_idx = str(idx)

            # Handle both dictionary and list formats
            item = data[str_idx] if isinstance(data, dict) else data[idx]

            # Basic text cleaning
            text = item['data'].strip()
            text = ' '.join(text.split())  # Normalize whitespace
            text = text.encode('ascii', 'ignore').decode('ascii')  # Remove problematic unicode

            # Store processed data
            texts.append(text)
            page_nums.append(item['page_number'])
            urls.append(item['URL'])
            promise_labels.append(item['promise_status'])
            evidence_labels.append(item['evidence_status'])

        logger.info(f"Loaded {len(texts)} samples")
        return texts, page_nums, urls, promise_labels, evidence_labels

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def main():
    try:
        # Set seeds for reproducibility
        set_seed(42)

        # Set device
        device = get_device()

        # Load and prepare data
        texts, page_nums, urls, promise_labels, evidence_labels = prepare_data(
            '../../data/english_train_augmented.json')

        # Split data with stratification
        train_idx, val_idx = train_test_split(
            range(len(texts)),
            test_size=0.1,
            random_state=42,
            stratify=[f"{p}_{e}" for p, e in zip(promise_labels, evidence_labels)]
        )

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        model = ESGModel(dropout=0.1)

        # Create datasets
        train_dataset = ESGDataset(
            [texts[i] for i in train_idx],
            [page_nums[i] for i in train_idx],
            [urls[i] for i in train_idx],
            [promise_labels[i] for i in train_idx],
            [evidence_labels[i] for i in train_idx],
            tokenizer,
            augment=True
        )

        val_dataset = ESGDataset(
            [texts[i] for i in val_idx],
            [page_nums[i] for i in val_idx],
            [urls[i] for i in val_idx],
            [promise_labels[i] for i in val_idx],
            [evidence_labels[i] for i in val_idx],
            tokenizer,
            augment=False
        )

        # Reduce batch size further and increase gradient accumulation
        batch_size = 1  # Keep as is
        gradient_accumulation_steps = 8  # Increase from 4
        max_length = 128  # Reduce from 256 to fit in memory

        # Modify data loader settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Reduce from 1
            pin_memory=True,  # Enable if using CUDA
            persistent_workers=False  # Disable to save memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True
        )

        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            device=device,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.2,
            gradient_accumulation_steps=2
        )

        # Load best model for predictions
        model.load_state_dict(torch.load('../../data/best_model.pt'))
        logger.info("Best model loaded for predictions")

        # Generate predictions
        predict_and_save(
            model=model,
            test_file='../../test/english_submission_file.json',
            output_file='../../test/predictions_ensemble.json',
            device=device
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        try_empty_cache()


if __name__ == "__main__":
    main()