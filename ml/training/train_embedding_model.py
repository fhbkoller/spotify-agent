import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.datasets import NoDuplicatesDataLoader
import logging
import os
import torch

# --- Configuration ---
TRAIN_DATA_FILE = "data/training_data.csv"
BASE_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
OUTPUT_MODEL_PATH = "data/models/custom-mxbai-embed-large-music" 

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    # --- 1. Load Data ---
    logger.info(f"Loading training data from {TRAIN_DATA_FILE}...")
    try:
        df = pd.read_csv(TRAIN_DATA_FILE)
        df.dropna(subset=['anchor', 'positive', 'negative'], inplace=True)
    except FileNotFoundError:
        logger.error(f"Error: {TRAIN_DATA_FILE} not found. Please run prepare_training_data.py first.")
        return
    if df.empty:
        logger.error("No data found in CSV. Exiting.")
        return

    # --- 2. Format Data ---
    logger.info(f"Formatting {len(df)} triplets into InputExamples...")
    train_examples = []
    for _, row in df.iterrows():
        example = InputExample(texts=[row['anchor'], row['positive'], row['negative']])
        train_examples.append(example)

    # --- 3. Check for GPU (CUDA) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"PyTorch has detected device: {device}")
    if device == 'cpu':
        logger.warning("WARNING: CUDA (GPU) not available. Training will be very slow on CPU.")

    # --- 4. Load Base Model ---
    logger.info(f"Loading base model: {BASE_MODEL}...")
    try:
        word_embedding_model = models.Transformer(
            BASE_MODEL, 
            model_args={"trust_remote_code": True}
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], 
            device=device
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # --- 5. Define Dataloader ---
    logger.info(f"Creating DataLoader with batch size {BATCH_SIZE}...")
    # *** THE FIX: Removed the 'shuffle=True' argument ***
    train_dataloader = NoDuplicatesDataLoader(
        train_examples,
        batch_size=BATCH_SIZE
    )

    # --- 6. Define Loss Function ---
    logger.info("Defining TripletLoss function...")
    train_loss = losses.TripletLoss(model=model)

    # --- 7. Train (Fine-Tune) the Model ---
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
    
    logger.info(f"--- Starting Model Training ---")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Warmup Steps: {warmup_steps}")
    logger.info(f"Output Path: {OUTPUT_MODEL_PATH}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True,
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=1
    )
    
    logger.info(f"--- Training Complete ---")
    logger.info(f"Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()