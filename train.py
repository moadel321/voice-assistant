import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import evaluate
import librosa
import logging
import psutil
import GPUtil
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
VALIDATION_FREQUENCY = 100
GRADIENT_ACCUMULATION_STEPS = 4
MAX_AUDIO_LENGTH = 30 * 16000  # 30 seconds at 16kHz

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logging.info(f"CPU Memory: {process.memory_info().rss / 1e9:.2f} GB")
    if torch.cuda.is_available():
        logging.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class WhisperDataset(Dataset):
    def __init__(self, df, processor, max_length=30*16000):  # 30 seconds at 16kHz
        self.df = df
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        text = row['sentence']  # Ensure this matches your DataFrame column name

        # Load and preprocess audio using librosa
        audio, sr = librosa.load(audio_path, sr=16000)  # Force 16kHz sample rate
        
        # Pad or truncate audio
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = np.zeros(self.max_length - len(audio))
            audio = np.concatenate((audio, padding))

        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

        # Tokenize text
        labels = self.processor(text=text, return_tensors="pt").input_ids

        return {"input_features": input_features.squeeze(), "labels": labels.squeeze()}

def load_dataset(metadata_file, audio_dir):
    df = pd.read_json(metadata_file, lines=True)
    df['audio_path'] = df['file_path'].apply(lambda x: f"{audio_dir}/{os.path.basename(x)}")
    return df

def collate_fn(batch):
    input_features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad input_features
    input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
    
    # Pad labels
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {"input_features": input_features, "labels": labels}

def compute_metrics(pred_str, label_str):
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def validate(model, dataloader, device, processor):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(input_features, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
            all_labels.extend(processor.batch_decode(labels, skip_special_tokens=True))
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    
    return avg_loss, metrics["wer"]

def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    return rank, local_rank, world_size

def train(local_rank, world_size):
    try:
        rank, local_rank, setup_world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")

        logging.info(f"Process {rank} (local_rank: {local_rank}) is starting.")

        logging.info("Starting training setup")
        log_memory_usage()

        # Load model
        logging.info("Loading model")
        model_name = "openai/whisper-large-v3"
        start_time = time.time()
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model = DDP(model, device_ids=[local_rank])
        logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        log_memory_usage()

        # Enable gradient checkpointing to save memory
        model.module.gradient_checkpointing_enable()
        processor = WhisperProcessor.from_pretrained(model_name)

        # Prepare datasets
        logging.info("Preparing datasets")
        df = load_dataset("metadata.jsonl", "audio_files")
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        logging.info(f"Split dataset: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

        train_dataset = WhisperDataset(train_df, processor, max_length=MAX_AUDIO_LENGTH)
        val_dataset = WhisperDataset(val_df, processor, max_length=MAX_AUDIO_LENGTH)
        test_dataset = WhisperDataset(test_df, processor, max_length=MAX_AUDIO_LENGTH)
        logging.info(f"Datasets prepared in {time.time() - start_time:.2f} seconds")
        log_memory_usage()

        # Use DistributedSampler
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        logging.info("Creating data loaders")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        scaler = torch.cuda.amp.GradScaler()
        best_val_loss = float('inf')

        logging.info("Starting training loop")
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_sampler.set_epoch(epoch)  # Shuffle data each epoch
            total_train_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training", disable=rank != 0)):
                input_features = batch["input_features"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_features, labels=labels)
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                
                total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if (step + 1) % VALIDATION_FREQUENCY == 0 and rank == 0:
                    val_loss, val_wer = validate(model.module, val_dataloader, device, processor)
                    logging.info(f"Step {step+1}, Validation Loss: {val_loss:.4f}, Validation WER: {val_wer:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.module.state_dict(), 'best_whisper_model.pth')
                        logging.info("Saved best model!")
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            if rank == 0:
                val_loss, val_wer = validate(model.module, val_dataloader, device, processor)
                
                logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
                logging.info(f"Average Training Loss: {avg_train_loss:.4f}")
                logging.info(f"Validation Loss: {val_loss:.4f}")
                logging.info(f"Validation WER: {val_wer:.4f}")
                
                scheduler.step(val_loss)
        
        if rank == 0:
            logging.info("Training completed!")
            logging.info("Evaluating on test set...")
            test_loss, test_wer = validate(model.module, test_dataloader, device, processor)
            logging.info(f"Test Loss: {test_loss:.4f}")
            logging.info(f"Test WER: {test_wer:.4f}")

    except Exception as e:
        logging.error(f"Encountered an error on rank {rank}: {str(e)}")
        logging.error(f"Error traceback: {traceback.format_exc()}")
        raise e
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(local_rank, world_size)
