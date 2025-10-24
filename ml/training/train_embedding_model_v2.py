import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os

# --- Configuration ---
TRAIN_DATA_FILE = "data/training_data_v2_augmented.csv" # v3 augmented data
MODEL_OUTPUT_PATH = "data/models/v2_1_pure_audio_model.pth" # Overwriting
MODEL_DIR = "data/models"

# Feature definitions
NUM_NUMERICAL_FEATURES = 9
NUM_CATEGORICAL_FEATURES = 2
CATEGORICAL_CARDINALITIES = [12, 2]

# --- Hyperparameters ---
EMBEDDING_DIM = 128
CAT_EMBEDDING_DIM = 8
BATCH_SIZE = 64
# *** CHANGED: Reduced epochs, 150 was too long for unstable run
EPOCHS = 100
# *** CHANGED: Reduce LR to prevent thrashing
LEARNING_RATE = 1e-4
# *** CHANGED: Revert to easier margin
TRIPLET_MARGIN = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- 1. Custom Dataset (Unchanged) ---
class SongVibeDataset(Dataset):
    def __init__(self, csv_file):
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            logger.error(f"Data file not found: {csv_file}")
            raise

        self.anchor_cols = [f'anchor_feat_{i}' for i in range(11)]
        self.pos_cols = [f'positive_feat_{i}' for i in range(11)]
        self.neg_cols = [f'negative_feat_{i}' for i in range(11)]

        self.num_indices = list(range(NUM_NUMERICAL_FEATURES))
        self.cat_indices = list(range(NUM_NUMERICAL_FEATURES,
                                     NUM_NUMERICAL_FEATURES + NUM_CATEGORICAL_FEATURES))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        anchor_all = row[self.anchor_cols].values.astype('float32')
        pos_all = row[self.pos_cols].values.astype('float32')
        neg_all = row[self.neg_cols].values.astype('float32')

        anchor_num = torch.tensor(anchor_all[self.num_indices], dtype=torch.float32)
        anchor_cat = torch.tensor(anchor_all[self.cat_indices], dtype=torch.long)

        pos_num = torch.tensor(pos_all[self.num_indices], dtype=torch.float32)
        pos_cat = torch.tensor(pos_all[self.cat_indices], dtype=torch.long)

        neg_num = torch.tensor(neg_all[self.num_indices], dtype=torch.float32)
        neg_cat = torch.tensor(neg_all[self.cat_indices], dtype=torch.long)

        return (anchor_num, anchor_cat), (pos_num, pos_cat), (neg_num, neg_cat)

# --- 2. The Two-Tower Model (Leaky ReLU Gated) ---
class SongVibeModel(nn.Module):
    def __init__(self, num_numerical_in, cat_cardinalities, cat_embed_dim, out_dim):
        super(SongVibeModel, self).__init__()

        self.numerical_tower = nn.Sequential(
            nn.Linear(num_numerical_in, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, cat_embed_dim) for cardinality in cat_cardinalities
        ])

        total_cat_embed_dim = len(cat_cardinalities) * cat_embed_dim

        self.categorical_tower = nn.Sequential(
            nn.Linear(total_cat_embed_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1)
        )

        combined_dim = 32

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_num, x_cat):
        out_num = self.numerical_tower(x_num)

        cat_embeds = []
        for i, embed_layer in enumerate(self.categorical_embeddings):
            cat_embeds.append(embed_layer(x_cat[:, i]))

        out_cat_flat = torch.cat(cat_embeds, dim=1)
        out_cat = self.categorical_tower(out_cat_flat)

        combined = out_num * out_cat

        output_embedding = self.head(combined)
        output_embedding = nn.functional.normalize(output_embedding, p=2, dim=1)

        return output_embedding

# --- 3. Training Function ---
def train_model():
    logger.info(f"Using device: {DEVICE}")

    logger.info(f"Loading AUGMENTED v3 dataset from {TRAIN_DATA_FILE}...")
    try:
        dataset = SongVibeDataset(TRAIN_DATA_FILE)
    except Exception as e:
        logger.error(f"Failed to load dataset. Exiting.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2
    )

    logger.info("Initializing Leaky ReLU GATED v2.1 model...")
    model = SongVibeModel(
        num_numerical_in=NUM_NUMERICAL_FEATURES,
        cat_cardinalities=CATEGORICAL_CARDINALITIES,
        cat_embed_dim=CAT_EMBEDDING_DIM,
        out_dim=EMBEDDING_DIM
    ).to(DEVICE)

    loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    logger.info(f"--- Starting Training (Stabilized Run) ---")
    logger.info(f"  Total Triplets: {len(dataset)}")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  Triplet Margin: {TRIPLET_MARGIN}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for (anchor_num, anchor_cat), (pos_num, pos_cat), (neg_num, neg_cat) in progress_bar:

            anchor_num, anchor_cat = anchor_num.to(DEVICE), anchor_cat.to(DEVICE)
            pos_num, pos_cat = pos_num.to(DEVICE), pos_cat.to(DEVICE)
            neg_num, neg_cat = neg_num.to(DEVICE), neg_cat.to(DEVICE)

            anchor_emb = model(anchor_num, anchor_cat)
            pos_emb = model(pos_num, pos_cat)
            neg_emb = model(neg_num, neg_cat)

            loss = loss_fn(anchor_emb, pos_emb, neg_emb)

            # Check for NaN loss, which indicates instability
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch {epoch+1}. Stopping training.")
                return # Stop training if it becomes unstable

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} completed. Average Loss: {avg_loss:.6f}")

    logger.info("Training complete.")
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info(f"Saving trained model state_dict to {MODEL_OUTPUT_PATH}...")
    try:
        torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    train_model()