from sentence_transformers import SentenceTransformer, util, models
import torch

# --- Model Configuration ---
BASE_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
FINETUNED_MODEL = "data/models/custom-mxbai-embed-large-music"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Test Suite Prompts ---

# [Test 1: Sanity Check] - The original test
T1_SONG_1 = "Track: War Pigs. Genres: rock, heavy metal. Features: acousticness=0.005, danceability=0.3, energy=0.7, instrumentalness=0.0, liveness=0.2, loudness=-8.0, speechiness=0.05, valence=0.3, tempo=120.00, key=D, mode=Minor."
T1_SONG_2 = "Track: Here Comes The Sun. Genres: rock, pop. Features: acousticness=0.6, danceability=0.5, energy=0.5, instrumentalness=0.001, liveness=0.1, loudness=-10.0, speechiness=0.03, valence=0.8, tempo=129.00, key=A, mode=Major."
T1_SONG_3 = "Track: Sandstorm. Genres: techno, trance. Features: acousticness=0.001, danceability=0.7, energy=0.9, instrumentalness=0.8, liveness=0.3, loudness=-5.0, speechiness=0.07, valence=0.5, tempo=136.00, key=G, mode=Minor."

# [Test 2: Mode Sensitivity] - Did it learn "Minor" vs "Major"?
T2_SONG_BASE = "Track: Generic Pop Song. Genres: pop, electronic. Features: acousticness=0.1, danceability=0.7, energy=0.8, instrumentalness=0.0, liveness=0.1, loudness=-6.0, speechiness=0.05, valence=0.6, tempo=120.00, key=C"
T2_SONG_MINOR = f"{T2_SONG_BASE}, mode=Minor."
T2_SONG_MAJOR = f"{T2_SONG_BASE}, mode=Major."

# [Test 3: Genre-Feature Connection] - Did it connect numbers to words?
T3_FEATURES_TECHNO = "Features: acousticness=0.01, danceability=0.8, energy=0.9, instrumentalness=0.8, liveness=0.3, loudness=-7.0, speechiness=0.05, valence=0.4, tempo=140.00, key=G, mode=Minor."
T3_GENRE_TECHNO = "Genres: techno, trance, electronic."
T3_GENRE_FOLK = "Genres: folk, acoustic, singer-songwriter."

# [Test 4: Numerical Sensitivity] - Is it more sensitive to feature changes?
T4_SONG_LOW_VIBE = "Track: Chillhop Track. Genres: lofi. Features: acousticness=0.8, danceability=0.6, energy=0.3, instrumentalness=0.7, liveness=0.1, loudness=-12.0, speechiness=0.04, valence=0.3, tempo=85.00, key=A, mode=Minor."
T4_SONG_HIGH_VIBE = "Track: Workout Track. Genres: edm. Features: acousticness=0.05, danceability=0.8, energy=0.9, instrumentalness=0.1, liveness=0.3, loudness=-5.0, speechiness=0.08, valence=0.7, tempo=128.00, key=G, mode=Major."


def load_model(model_name_or_path, is_base_model=False):
    """Helper function to load models consistently."""
    print(f"Loading {model_name_or_path}...")
    if is_base_model:
        # Base model requires explicit component loading
        word_embedding_model = models.Transformer(
            model_name_or_path, 
            model_args={"trust_remote_code": True}
        )
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=DEVICE)
    else:
        # Our fine-tuned model is saved in the standard format
        model = SentenceTransformer(model_name_or_path, device=DEVICE)
    return model

def run_sanity_check(base_model, tuned_model):
    """[Test 1] Simple check to see if the model's logic is reasonable."""
    print("\n" + "="*50)
    print("--- Test 1: Sanity Check (War Pigs vs. Sandstorm) ---")
    print("="*50)
    print("Hypothesis: Tuned model should see high-energy 'War Pigs' (rock)")
    print("            as more similar to 'Sandstorm' (techno) than to the")
    print("            acoustic, major-key 'Here Comes The Sun'.")
    
    prompts = [T1_SONG_1, T1_SONG_2, T1_SONG_3]
    base_emb = base_model.encode(prompts, convert_to_tensor=True)
    tuned_emb = tuned_model.encode(prompts, convert_to_tensor=True)

    base_sim_1v2 = util.cos_sim(base_emb[0], base_emb[1]).item()
    base_sim_1v3 = util.cos_sim(base_emb[0], base_emb[2]).item()
    
    tuned_sim_1v2 = util.cos_sim(tuned_emb[0], tuned_emb[1]).item()
    tuned_sim_1v3 = util.cos_sim(tuned_emb[0], tuned_emb[2]).item()

    print(f"\nBase Model:")
    print(f"  Sim(War Pigs, Here Comes Sun): {base_sim_1v2:.4f}")
    print(f"  Sim(War Pigs, Sandstorm):      {base_sim_1v3:.4f}")
    
    print(f"\nFine-Tuned Model:")
    print(f"  Sim(War Pigs, Here Comes Sun): {tuned_sim_1v2:.4f}")
    print(f"  Sim(War Pigs, Sandstorm):      {tuned_sim_1v3:.4f}")

    if tuned_sim_1v3 > tuned_sim_1v2:
        print("\n[Test 1 SUCCESS]: Tuned model correctly found Song 1 and 3 to be more similar.")
    else:
        print("\n[Test 1 FAILURE]: Tuned model failed this check.")

def run_mode_sensitivity_test(base_model, tuned_model):
    """[Test 2] Checks if the model learned 'Minor' vs 'Major'."""
    print("\n" + "="*50)
    print("--- Test 2: Mode Sensitivity Test ---")
    print("="*50)
    print("Hypothesis: Base model sees '...mode=Minor' and '...mode=Major'")
    print("            as ~99% similar. The Tuned model should have learned")
    print("            they are significantly different.")
    
    prompts = [T2_SONG_MINOR, T2_SONG_MAJOR]
    base_emb = base_model.encode(prompts, convert_to_tensor=True)
    tuned_emb = tuned_model.encode(prompts, convert_to_tensor=True)

    base_sim = util.cos_sim(base_emb[0], base_emb[1]).item()
    tuned_sim = util.cos_sim(tuned_emb[0], tuned_emb[1]).item()

    print(f"\nSimilarity between '...mode=Minor' and '...mode=Major':")
    print(f"  Base Model:       {base_sim:.4f}")
    print(f"  Fine-Tuned Model: {tuned_sim:.4f}")

    # Assertion: The similarity should drop by at least 5% (0.05)
    sim_drop = base_sim - tuned_sim
    if tuned_sim < 0.95 and sim_drop > 0.05:
        print(f"\n[Test 2 SUCCESS]: Similarity dropped by {sim_drop:.4f}. The model learned the difference.")
    else:
        print(f"\n[Test 2 FAILURE]: Model is still too sensitive to text. Similarity drop was only {sim_drop:.4f}.")

def run_genre_feature_test(base_model, tuned_model):
    """[Test 3] Checks if the model connected numerical features to genre words."""
    print("\n" + "="*50)
    print("--- Test 3: Genre-Feature Connection Test ---")
    print("="*50)
    print("Hypothesis: Tuned model should connect 'techno features' to")
    print("            the *word* 'techno', while the base model will not.")
    
    prompts = [T3_FEATURES_TECHNO, T3_GENRE_TECHNO, T3_GENRE_FOLK]
    base_emb = base_model.encode(prompts, convert_to_tensor=True)
    tuned_emb = tuned_model.encode(prompts, convert_to_tensor=True)

    base_sim_techno = util.cos_sim(base_emb[0], base_emb[1]).item()
    base_sim_folk = util.cos_sim(base_emb[0], base_emb[2]).item()

    tuned_sim_techno = util.cos_sim(tuned_emb[0], tuned_emb[1]).item()
    tuned_sim_folk = util.cos_sim(tuned_emb[0], tuned_emb[2]).item()

    print("\nSimilarity(Techno Features, Techno Genre):")
    print(f"  Base Model:       {base_sim_techno:.4f}")
    print(f"  Fine-Tuned Model: {tuned_sim_techno:.4f}")
    
    print("\nSimilarity(Techno Features, Folk Genre):")
    print(f"  Base Model:       {base_sim_folk:.4f}")
    print(f"  Fine-Tuned Model: {tuned_sim_folk:.4f}")

    # Assertions:
    # 1. The connection to 'techno' should be stronger.
    # 2. The connection to 'folk' should be weaker.
    # 3. The tuned model must score 'techno' higher than 'folk'.
    if (tuned_sim_techno > base_sim_techno + 0.05) and \
       (tuned_sim_folk < base_sim_folk + 0.05) and \
       (tuned_sim_techno > tuned_sim_folk):
        print(f"\n[Test 3 SUCCESS]: Model correctly associated features with 'techno' genre.")
    else:
        print(f"\n[Test 3 FAILURE]: Model did not learn the genre-feature connection.")
        
def run_numerical_sensitivity_test(base_model, tuned_model):
    """[Test 4] Checks if the model is more sensitive to musical distance."""
    print("\n" + "="*50)
    print("--- Test 4: Numerical Sensitivity Test ---")
    print("="*50)
    print("Hypothesis: Tuned model should see 'Low Vibe' and 'High Vibe'")
    print("            as much more *dissimilar* (lower score) than")
    print("            the base model, which just sees similar text structure.")
    
    prompts = [T4_SONG_LOW_VIBE, T4_SONG_HIGH_VIBE]
    base_emb = base_model.encode(prompts, convert_to_tensor=True)
    tuned_emb = tuned_model.encode(prompts, convert_to_tensor=True)

    base_sim = util.cos_sim(base_emb[0], base_emb[1]).item()
    tuned_sim = util.cos_sim(tuned_emb[0], tuned_emb[1]).item()

    print(f"\nSimilarity between 'Low Vibe' and 'High Vibe' songs:")
    print(f"  Base Model:       {base_sim:.4f}")
    print(f"  Fine-Tuned Model: {tuned_sim:.4f}")

    # Assertion: The tuned model should place them further apart.
    if tuned_sim < base_sim - 0.05:
        print(f"\n[Test 4 SUCCESS]: Tuned model created {base_sim - tuned_sim:.4f} more distance.")
    else:
        print(f"\n[Test 4 FAILURE]: Tuned model did not create significant new distance.")

def main():
    """Runs the full evaluation suite."""
    print(f"Using device: {DEVICE}")
    
    try:
        base_model = load_model(BASE_MODEL, is_base_model=True)
        tuned_model = load_model(FINETUNED_MODEL, is_base_model=False)
    except Exception as e:
        print(f"\n--- CRITICAL ERROR ---")
        print(f"Failed to load models. Error: {e}")
        print(f"Make sure '{FINETUNED_MODEL}' exists and you ran train_embedding_model.py.")
        return

    # Run all tests
    run_sanity_check(base_model, tuned_model)
    run_mode_sensitivity_test(base_model, tuned_model)
    run_genre_feature_test(base_model, tuned_model)
    run_numerical_sensitivity_test(base_model, tuned_model)
    
    print("\n" + "="*50)
    print("--- Test Suite Complete ---")
    print("="*50)

if __name__ == "__main__":
    main()