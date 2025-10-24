import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import util
import logging

# --- Model & File Configuration ---
MODEL_PATH = "data/models/v2_1_pure_audio_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Model Architecture (must match training script) ---
NUM_NUMERICAL_FEATURES = 9
NUM_CATEGORICAL_FEATURES = 2
CATEGORICAL_CARDINALITIES = [12, 2]
CAT_EMBEDDING_DIM = 8 
EMBEDDING_DIM = 128

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- 1. Model Class Definition (*** Leaky ReLU FIX ***) ---
class SongVibeModel(nn.Module):
    def __init__(self, num_numerical_in, cat_cardinalities, cat_embed_dim, out_dim):
        super(SongVibeModel, self).__init__()
        
        self.numerical_tower = nn.Sequential(
            nn.Linear(num_numerical_in, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1), # *** CHANGED
            nn.Linear(64, 32)
        )
        
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, cat_embed_dim) for cardinality in cat_cardinalities
        ])
        
        total_cat_embed_dim = len(cat_cardinalities) * cat_embed_dim
        
        self.categorical_tower = nn.Sequential(
            nn.Linear(total_cat_embed_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1) # *** CHANGED
        )

        combined_dim = 32
        
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1), # *** CHANGED
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

# --- 2. Test Helper Functions (Unchanged) ---
def load_v2_model(model_path):
    logger.info("Loading Leaky ReLU GATED v2.1 model...")
    try:
        model = SongVibeModel(
            num_numerical_in=NUM_NUMERICAL_FEATURES,
            cat_cardinalities=CATEGORICAL_CARDINALITIES,
            cat_embed_dim=CAT_EMBEDDING_DIM,
            out_dim=EMBEDDING_DIM
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() 
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"CRITICAL: Model file not found at {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def get_embedding(model, scaler, raw_vector: list):
    raw_vector_np = np.array(raw_vector).reshape(1, -1)
    
    raw_num = raw_vector_np[:, :NUM_NUMERICAL_FEATURES]
    raw_cat = raw_vector_np[:, NUM_NUMERICAL_FEATURES:]

    scaled_num = scaler.transform(raw_num)
    
    x_num = torch.tensor(scaled_num, dtype=torch.float32).to(DEVICE)
    x_cat = torch.tensor(raw_cat, dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        embedding = model(x_num, x_cat)
        
    return embedding

# --- 3. Test Definitions (Unchanged) ---
def run_test_2_mode_sensitivity(model):
    print("\n--- [Test 2: Mode Sensitivity] ---")
    
    base_song = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 6, 0] # 6 = F#
    minor_song = base_song[:] 
    minor_song[-1] = 0 # 0 = Minor
    major_song = base_song[:] 
    major_song[-1] = 1 # 1 = Major
    
    scaler = MinMaxScaler()
    num_data_to_fit = np.array([minor_song, major_song])[:, :NUM_NUMERICAL_FEATURES]
    scaler.fit(num_data_to_fit)
    
    minor_emb = get_embedding(model, scaler, minor_song)
    major_emb = get_embedding(model, scaler, major_song)
    
    similarity = util.cos_sim(minor_emb, major_emb).item()
    
    print(f"Similarity between 'F# Minor' and 'F# Major' (all else equal):")
    print(f"  v2.1 (Leaky) Model: {similarity:.4f}")
    
    if similarity < 0.95:
        print(f"[Test 2 SUCCESS]: Model is sensitive to 'mode'. Distance: {1.0 - similarity:.4f}")
    else:
        print(f"[Test 2 FAILURE]: Model is IGNORING 'mode'.")
    print("---------------------------------")

def run_test_4_numerical_sensitivity(model):
    print("\n--- [Test 4: Numerical Sensitivity] ---")
    
    low_vibe_song = [0.0, 0.1, 0.1, 0.0, 0.1, 0.5, 0.05, 0.1, 0.5, 0, 1] # C Major
    high_vibe_song = [0.0, 0.9, 0.9, 0.0, 0.1, 0.5, 0.05, 0.9, 0.5, 0, 1] # C Major
    
    scaler = MinMaxScaler()
    num_data_to_fit = np.array([low_vibe_song, high_vibe_song])[:, :NUM_NUMERICAL_FEATURES]
    scaler.fit(num_data_to_fit)
    
    low_vibe_emb = get_embedding(model, scaler, low_vibe_song)
    high_vibe_emb = get_embedding(model, scaler, high_vibe_song)

    similarity = util.cos_sim(low_vibe_emb, high_vibe_emb).item()

    print(f"Similarity between 'Low Vibe' and 'High Vibe' songs:")
    print(f"  v2.1 (Leaky) Model: {similarity:.4f}")
    
    if similarity < 0.95:
        print(f"[Test 4 SUCCESS]: Model is sensitive to numerical features. Distance: {1.0 - similarity:.4f}")
    else:
        print(f"[Test 4 FAILURE]: Model is IGNORING numerical features.")
    print("------------------------------------")

# --- 4. Main Execution ---
def main():
    logger.info(f"Using device: {DEVICE}")
    model = load_v2_model(MODEL_PATH)
    
    if model:
        logger.info("--- Starting v2.1 (Leaky) Model Validation ---")
        run_test_2_mode_sensitivity(model)
        run_test_4_numerical_sensitivity(model)
        logger.info("--- Validation Complete ---")

if __name__ == "__main__":
    main()