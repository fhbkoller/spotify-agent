# The Spotify Agent Roadmap

This document outlines the strategic high-level plan for developing the Spotify Agent, from validating the current model to integrating a specialized v2.2 model and enabling stateful context awareness.

## Phase 1: V2 Model Final Validation

* **Goal:** To establish a final, definitive performance baseline for the current V2 (`Leaky Gated`) model and `v2_augmented` dataset. This is a "closing the books" phase to justify our pivot.
* **Key Actions:**
    1.  Execute a single, final training run using the current best model candidate, data, and the one identified high-impact tweak (reduced triplet margin).
    2.  Perform a comprehensive analysis of this model's performance using all existing test scripts (`test_model_v2.py`, `test_model_v2_training_fit.py`).
* **Deliverable:**
    * A final **V2 Model Performance Report** (e.g., loss curves, "Goodness of Fit" score, and pass/fail on sensitivity tests). This report will serve as the benchmark to beat.

## Phase 2: Strategic Data Pivot (Breadth & Specificity)

* **Goal:** To engineer a new, fundamentally superior training dataset that will teach the model both broad, real-world musical relationships and the fine-grained "vibe" nuances we require.
* **Key Actions:**
    1.  **Real-World Data Curation (Breadth):** Identify and process a large-scale, public music dataset (e.g., FMA). The focus is on extracting audio features and existing human labels (like genre) to create a massive set of "ecologically valid" training triplets. This dataset teaches the model the general "shape" of the music universe.
    2.  **Synthetic Data Generation (Depth & Specificity):** Engineer a new, comprehensive synthetic dataset based on the principles in `Music Embedding Training Data Generation.md`. This dataset will be the **primary source for teaching specificity**. Its design must encompass:
        * **Hard Positives/Negatives:** Systematically generate triplets that test the boundaries of each individual audio feature.
        * **Genre & Mood Simulation:** Create statistically-grounded "profiles" for various genres and moods, generating triplets that force the model to learn these complex, multi-feature combinations.
* **Deliverables:**
    1.  A new, large-scale **Real-World Triplet Dataset** (e.g., `fma_triplets.csv`).
    2.  A new, comprehensive **Synthetic Specificity Dataset** (e.g., `synthetic_specificity_v1.csv`) that includes all designated hard cases, genre simulations, and mood profiles.
    3.  A new data pipeline module (`ml/data_pipeline`) to house the scripts that generate these datasets.

## Phase 3: Model v2.2 Specialization (Multi-Task Architecture)

* **Goal:** To develop a new v2.2 model architecture that produces a significantly more robust and semantically rich song embedding by leveraging our new, high-quality datasets.
* **Key Actions:**
    1.  **Define a Multi-Task Learning (MTL) Architecture:** Evolve the model to have a shared "trunk" (which produces the embedding) and two specialized "heads":
        * **Head 1 (Triplet Loss):** To learn vibe/similarity.
        * **Head 2 (Classification Loss):** To simultaneously predict a song's genre (using the labels from our Phase 2 real-world dataset).
    2.  **Implement a Two-Stage Training Strategy:**
        * **Stage 1 (Pre-training):** Train the MTL model on the large, real-world dataset.
        * **Stage 2 (Fine-tuning):** Fine-tune the pre-trained model on our new `Synthetic Specificity Dataset` to master the subtle "vibe" nuances.
* **Deliverable:**
    * A new, trained **V2.2 Multi-Task Model** (e.g., `embedding_model_v2.2.pth`) that demonstrates superior performance on all sensitivity tests.

## Phase 4: Agent AI Integration (Vibe Model & Stateful Context)

* **Goal:** To integrate *both* the custom AI vibe model and the new session memory into the core agent, evolving it from a stateless to a stateful system.
* **Key Actions:**
    1.  **Integrate V2.2 Vibe Model:**
        * Define a `VibeModelService` (e.g., in `src/services/vibe_model.py`).
        * This service will load the `embedding_model_v2.2.pth` file at startup.
        * It will expose a method: `get_embedding(features: dict) -> list[float]` that handles all PyTorch preprocessing and inference logic directly within the application.
    2.  **Define `SessionContext` Object:**
        * Design a new, in-memory object responsible for tracking the state of a single listening session.
    3.  **Integrate Context Storage:**
        * The `SessionContext` will be responsible for maintaining two key pieces of information:
            * **Recent Listening History:** (e.g., 'skipped', 'fully played').
            * **Session Sonic Profile:** (e.g., running average of `energy`, `valence`).
    4.  **Upgrade Agent Prompting:**
        * Modify the `_get_ai_score_multiplier_async` function to:
            * *Accept* the `SessionContext` as a parameter.
            * *Format* the history and sonic profile into new sections of the LLM prompt.
            * *Instruct* the LLM to use this new context to make more intelligent, trend-aware scoring decisions.
* **Deliverables:**
    1.  A new **`VibeModelService`** that serves our custom `.pth` model as a simple Python service.
    2.  A modified **Core Agent** (`src/core/agent.py`) that actively maintains and utilizes the `SessionContext`, resulting in AI-driven scores that are adaptive to the user's immediate behavior.
