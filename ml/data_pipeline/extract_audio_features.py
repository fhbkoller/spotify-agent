# ml/data_pipeline/extract_features_pipeline_concurrent.py

import sys
import time
import warnings
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, cpu_count
import os
import shutil
import glob
import random

# --- Force Single-Threaded CPU within each worker ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback  # <-- IMPORTED FOR LOGGING

import torch
import torchaudio
import torchaudio.functional as F

# --- Configuration ---
AUDIO_DIR = Path("ml/data/raw/mtg-jamendo/low_quality_audio")
LOOKUP_DIR = Path("ml/data/lookups")
OUTPUT_FEATURES_FILE = LOOKUP_DIR / "features.parquet"

TEMP_STAGE1_DIR = LOOKUP_DIR / "temp_stage1_gpu_out" # GPU workers write .npz here
TEMP_STAGE2_DIR = LOOKUP_DIR / "temp_stage2_cpu_out" # CPU workers write .parquet here

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 2500 # For final parquet files

# --- !! VOLATILE WORKER Configuration (maxtasksperchild) !! ---
N_ACTIVE_GPU_WORKERS = 5  
GPU_WORKER_BATCH_SIZE = 10 # maxtasksperchild

N_ACTIVE_CPU_WORKERS = 4 
CPU_WORKER_BATCH_SIZE = 5  # maxtasksperchild

N_RESERVED_CORES = N_ACTIVE_GPU_WORKERS + N_ACTIVE_CPU_WORKERS + 1
if cpu_count() <= N_RESERVED_CORES:
    print(f"Warning: Low core count. {cpu_count()} cores.")
    
QUEUE_MAX_SIZE = N_ACTIVE_CPU_WORKERS * 4 # Backpressure

warnings.filterwarnings('ignore')

# --- Librosa Key/Mode Helpers (Unchanged) ---
major_profile_np = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_profile_np = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
major_profile_np /= np.linalg.norm(major_profile_np)
minor_profile_np /= np.linalg.norm(minor_profile_np)
key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def estimate_key_mode_np(chroma_mean_np):
    if chroma_mean_np is None or len(chroma_mean_np) != 12: return 'C', 'major', 0.0
    major_corrs, minor_corrs = [], []
    for i in range(12):
        rolled_chroma = np.roll(chroma_mean_np, -i)
        if not isinstance(rolled_chroma, np.ndarray) or not isinstance(major_profile_np, np.ndarray): continue
        try:
            major_corrs.append(np.corrcoef(rolled_chroma, major_profile_np)[0, 1])
            minor_corrs.append(np.corrcoef(rolled_chroma, minor_profile_np)[0, 1])
        except ValueError:
             major_corrs.append(0.0)
             minor_corrs.append(0.0)
    if not major_corrs or not minor_corrs: return 'C', 'major', 0.0
    major_key_idx = np.argmax(major_corrs) if major_corrs else 0
    minor_key_idx = np.argmax(minor_corrs) if minor_corrs else 0
    major_max = major_corrs[major_key_idx] if major_corrs else 0.0
    minor_max = minor_corrs[minor_key_idx] if minor_corrs else 0.0
    major_key_idx = major_key_idx % len(key_names)
    minor_key_idx = minor_key_idx % len(key_names)
    if major_max >= minor_max:
        return key_names[major_key_idx], 'major', float(major_max)
    else:
        return key_names[minor_key_idx], 'minor', float(minor_max)

# --- Manual Spectral Flatness (Torch) ---
def spectral_flatness_torch(S_power_gpu, epsilon=1e-10):
    log_spec = torch.log(S_power_gpu + epsilon)
    geom_mean = torch.exp(torch.mean(log_spec, dim=0))
    arith_mean = torch.mean(S_power_gpu, dim=0)
    flatness = geom_mean / (arith_mean + epsilon)
    return flatness
    
# --- Manual Zero Crossing Rate (Torch) ---
def zero_crossing_rate_torch(waveform_gpu):
    zcr = torch.mean((waveform_gpu[:-1] * waveform_gpu[1:] < 0).float())
    return zcr

# --- === PIPELINE WORKER FUNCTIONS === ---

def gpu_worker(file_queue: Queue, cpu_work_queue: Queue, worker_id: int, temp_stage1_dir: Path):
    """
    Producer: Gets a file path, runs all fast GPU tasks,
    saves intermediate file to disk, and puts the *path* on the cpu_work_queue.
    """
    
    # --- FIX 1 (VRAM LEAK): Check for CUDA *inside* the worker, not in main. ---
    if not torch.cuda.is_available():
        print(f"GPU Worker {worker_id}: FATAL ERROR: CUDA not available.")
        sys.exit(1) # Exit this process, Pool will try to restart it
        
    worker_device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
    
    stft_transform = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0).to(worker_device)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128).to(worker_device)
    centroid_transform = torchaudio.transforms.SpectralCentroid(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH).to(worker_device)
    resampler_map = {}
    
    while True:
        # --- FIX 3 (Aggressive Cleanup): Define vars to clear ---
        waveform = y_gpu = S_gpu_power = S_gpu_mag = M_gpu = None
        rms_vec_gpu = centroid_gpu = flatness_gpu_vec = zcr_gpu = None
        
        try:
            audio_file = file_queue.get()
            if audio_file is None: 
                cpu_work_queue.put(None) # Pass sentinel
                break                    # Exit loop
            
            track_id = audio_file.stem.replace(".low", "")
            out_path = temp_stage1_dir / f"{track_id}.npz"
        
            waveform, sr = torchaudio.load(str(audio_file))
            waveform = waveform.to(worker_device)
            if sr != SAMPLE_RATE:
                if sr not in resampler_map:
                    resampler_map[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE).to(worker_device)
                waveform = resampler_map[sr](waveform)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            y_gpu = waveform.squeeze()
            if y_gpu.numel() == 0: continue

            S_gpu_power = stft_transform(y_gpu)
            S_gpu_mag = S_gpu_power.sqrt()
            M_gpu = mel_transform(y_gpu)
            rms_vec_gpu = S_gpu_power.mean(dim=0).sqrt()
            centroid_gpu = centroid_transform(S_gpu_power).squeeze()
            flatness_gpu_vec = spectral_flatness_torch(S_gpu_power)
            zcr_gpu = zero_crossing_rate_torch(y_gpu)

            features = {
                'track_id': track_id,
                'loudness': float(rms_vec_gpu.mean().item()),
                'energy': float(rms_vec_gpu.mean().item()),
                'acousticness': float(1.0 - (centroid_gpu.mean() / (SAMPLE_RATE / 2)).item()),
                'instrumentalness': float(1.0 - flatness_gpu_vec.mean().item()),
                'danceability': float(zcr_gpu.item()),
            }

            intermediate_data = {
                'features': features,
                'M_cpu': M_gpu.cpu().numpy(),
                'S_cpu_mag': S_gpu_mag.cpu().numpy(),
                'rms_vec_cpu': rms_vec_gpu.cpu().numpy(),
            }
            
            np.savez_compressed(out_path, **intermediate_data)
            cpu_work_queue.put(out_path) 

        except Exception as e:
            # --- FIX 2 (Silent Failure): ADD LOGGING! ---
            print(f"\n---!!! GPU WORKER ERROR (Track: {track_id}) !!!---")
            traceback.print_exc()
            print("--------------------------------------------------\n")
            continue
            
        finally:
            # --- FIX 3 (Aggressive Cleanup): Explicitly delete all tensors ---
            try:
                if waveform is not None: del waveform
                if y_gpu is not None: del y_gpu
                if S_gpu_power is not None: del S_gpu_power
                if S_gpu_mag is not None: del S_gpu_mag
                if M_gpu is not None: del M_gpu
                if rms_vec_gpu is not None: del rms_vec_gpu
                if centroid_gpu is not None: del centroid_gpu
                if flatness_gpu_vec is not None: del flatness_gpu_vec
                if zcr_gpu is not None: del zcr_gpu
                torch.cuda.empty_cache()
            except Exception:
                pass # Avoid crashing in the cleanup


def cpu_worker(cpu_work_queue: Queue, results_queue: Queue, worker_id: int):
    """
    Consumer: Gets an intermediate .npz file *path* from gpu_work_queue,
    loads it, runs all slow CPU-bound librosa tasks, deletes the .npz,
    and puts the final feature dict onto the results_queue.
    """
    
    while True:
        track_id = "UNKNOWN (in CPU worker)"
        try:
            intermediate_file_path = cpu_work_queue.get()
            if intermediate_file_path is None: 
                results_queue.put(None)  # Pass sentinel
                break                    # Exit loop

            with np.load(str(intermediate_file_path), allow_pickle=True) as data:
                features = data['features'].item()
                M_cpu = data['M_cpu']
                S_cpu_mag = data['S_cpu_mag']
                rms_vec_cpu = data['rms_vec_cpu']
            
            track_id = features.get('track_id', 'UNKNOWN')

            onset_env = librosa.onset.onset_strength(S=M_cpu, sr=SAMPLE_RATE)
            features['liveness'] = float(np.mean(onset_env))
            
            tempo_array = librosa.feature.tempo(onset_envelope=onset_env, sr=SAMPLE_RATE)
            features['tempo'] = float(tempo_array[0]) if tempo_array.size > 0 else 120.0
            
            chroma = librosa.feature.chroma_stft(S=S_cpu_mag, sr=SAMPLE_RATE)
            chroma_mean = np.mean(chroma, axis=1)
            key_str, mode_str, val_proxy = estimate_key_mode_np(chroma_mean)
            features['key'] = key_str
            features['mode'] = mode_str
            features['valence'] = float(val_proxy)

            rms_threshold = 0.01
            silent_frames = np.sum(rms_vec_cpu < rms_threshold)
            total_frames = len(rms_vec_cpu)
            silence_rate = silent_frames / total_frames if total_frames > 0 else 0.0
            features['speechiness'] = float(1.0 - silence_rate)
            
            if any(v is None for v in features.values()):
                continue

            results_queue.put(features)

        except Exception as e:
            # --- FIX 2 (Silent Failure): ADD LOGGING! ---
            print(f"\n---!!! CPU WORKER ERROR (Track: {track_id}) !!!---")
            traceback.print_exc()
            print("------------------------------------------------\n")
            pass
        
        finally:
            try:
                if 'intermediate_file_path' in locals() and intermediate_file_path:
                    os.remove(intermediate_file_path)
            except OSError:
                pass


def result_writer_process(
    results_queue: Queue, 
    total_files: int,
    temp_stage2_dir: Path, 
    batch_size: int, 
    n_cpu_workers: int
):
    """
    Writer: Gets final feature dicts from results_queue
    and saves them to parquet files in batches.
    Manages the master progress bar.
    """
    temp_stage2_dir.mkdir(parents=True, exist_ok=True)
    
    results_batch = []
    finished_cpu_workers = 0
    batch_num = 1
    
    existing_temp_files = sorted(temp_stage2_dir.glob("features_part_*.parquet"))
    files_processed_so_far = 0
    processed_track_ids = set()
    
    if existing_temp_files:
        print(f"Found {len(existing_temp_files)} existing batch files. Resuming...")
        try:
            for f in tqdm(existing_temp_files, desc="Scanning completed files", unit="file"):
                df_temp = pd.read_parquet(f)
                processed_track_ids.update(df_temp['track_id'].values)
            files_processed_so_far = len(processed_track_ids)
            batch_num = len(existing_temp_files) + 1
            print(f"Found {files_processed_so_far} tracks already processed.")
        except Exception as e:
            print(f"Warning: Could not read all temp files ({e}). Starting from scratch.")
            files_processed_so_far = 0
            
    pbar = tqdm(total=total_files, desc="Extracting Features (Pipeline)", unit="file", initial=files_processed_so_far)
    
    try:
        while finished_cpu_workers < n_cpu_workers: 
            result = results_queue.get()
            
            if result is None:
                finished_cpu_workers += 1
                continue
                
            if result['track_id'] not in processed_track_ids:
                results_batch.append(result)
                pbar.update(1)
            
            if len(results_batch) >= batch_size:
                temp_file_path = temp_stage2_dir / f"features_part_{batch_num:04d}.parquet"
                df_batch = pd.DataFrame(results_batch)
                df_batch.to_parquet(temp_file_path, engine='pyarrow') # This is where files are written
                results_batch = []
                batch_num += 1
    
    except Exception as e:
        print("\n---!!! RESULT WRITER ERROR !!!---")
        traceback.print_exc()
        print("---------------------------------\n")
    
    finally:
        pbar.close()
        
        if results_batch: # Save the final partial batch
            try:
                temp_file_path = temp_stage2_dir / f"features_part_{batch_num:04d}.parquet"
                df_batch = pd.DataFrame(results_batch)
                df_batch.to_parquet(temp_file_path, engine='pyarrow')
            except Exception as e:
                print("\n---!!! RESULT WRITER ERROR (FINAL BATCH) !!!---")
                traceback.print_exc()
            
        print("\n--- All workers finished. Consolidation process will now run. ---")


# --- run_consolidation_stage (Unchanged) ---
def run_consolidation_stage():
    """(Runs in Main Process) Cleans up and merges all parquet files."""
    print("\n--- STAGE 3: Consolidating Final Output ---")
    
    all_temp_files = sorted(list(TEMP_STAGE2_DIR.glob("features_part_*.parquet")))
    print(f"Found {len(all_temp_files)} final batch files to consolidate.")
    
    all_dfs = []
    if not all_temp_files:
        print("FATAL ERROR: No temporary batch files were created.")
        return

    for f in all_temp_files:
        try:
            all_dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Warning: Could not read {f}, skipping. Error: {e}")
    
    if not all_dfs:
        print("FATAL ERROR: No valid batch files could be read.")
        return

    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
    
    print(f"\nSuccessfully consolidated {len(df_final)} tracks.")
    
    # --- Final Data Cleaning ---
    print("Cleaning and typing final data...")
    df_final.set_index('track_id', inplace=True)
    
    def key_to_int(key_str):
        key_str = str(key_str).upper()
        key_map = {'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5,
                   'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11}
        return key_map.get(str(key_str), 0)
    df_final['key'] = df_final['key'].apply(key_to_int).astype(int)
    df_final['mode'] = df_final['mode'].apply(lambda x: 1 if str(x).lower() == 'major' else 0).astype(int)
    
    numerical_cols = [f for f in df_final.columns if f not in ['key', 'mode']]
    df_final[numerical_cols] = df_final[numerical_cols].astype(float)
    
    initial_count = len(df_final)
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.dropna(inplace=True)
    if len(df_final) < initial_count: print(f"Dropped {initial_count - len(df_final)} tracks due to NaN/Inf values.")
    
    final_columns = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'speechiness', 'valence', 'loudness', 'tempo',
        'key', 'mode'
    ]
    df_final = df_final.reindex(columns=final_columns)

    print(f"Saving {len(df_final)} tracks with 11 features to {OUTPUT_FEATURES_FILE}")
    OUTPUT_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_FEATURES_FILE, engine='pyarrow')
    
    try:
        shutil.rmtree(TEMP_STAGE1_DIR)
        shutil.rmtree(TEMP_STAGE2_DIR)
        print(f"Cleaned up temporary directories.")
    except Exception as e:
        print(f"Warning: Could not clean up temp directories: {e}")


# --- Main Orchestration Logic ---
def main():
    start_time = time.time()
    
    # --- FIX 1 (VRAM LEAK): REMOVED all CUDA checks from main. ---
        
    print(f"--- STARTING Volatile Multi-Process Pipeline (maxtasksperchild) ---")
    print(f"Active GPU Workers: {N_ACTIVE_GPU_WORKERS} (Batch Size: {GPU_WORKER_BATCH_SIZE})")
    print(f"Active CPU Workers: {N_ACTIVE_CPU_WORKERS} (Batch Size: {CPU_WORKER_BATCH_SIZE})")
    
    TEMP_STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    all_audio_files = list(AUDIO_DIR.glob("**/*.mp3"))
    total_files = len(all_audio_files)

    if not all_audio_files:
        print(f"FATAL ERROR: No audio files found in {AUDIO_DIR}")
        return
        
    print(f"Found {total_files} audio files to process.")

    manager = Manager()
    file_queue = manager.Queue()
    cpu_work_queue = manager.Queue(maxsize=QUEUE_MAX_SIZE)
    results_queue = manager.Queue()

    # --- Scan for already processed files (Fault Tolerance) ---
    processed_track_ids = set()
    temp_parquet_files = sorted(list(TEMP_STAGE2_DIR.glob("features_part_*.parquet")))
    if temp_parquet_files:
        print(f"Found {len(temp_parquet_files)} existing batch files. Resuming...")
        try:
            for f in tqdm(temp_parquet_files, desc="Scanning completed files"):
                df_temp = pd.read_parquet(f)
                processed_track_ids.update(df_temp['track_id'].values)
            print(f"Found {len(processed_track_ids)} tracks already processed. Skipping them.")
        except Exception as e:
            print(f"Warning: Could not read all temp files ({e}). Starting from scratch.")
            processed_track_ids = set()

    # --- 1. Start Writer Process ---
    writer_proc = Process(target=result_writer_process, args=(
        results_queue, total_files, 
        TEMP_STAGE2_DIR, BATCH_SIZE, N_ACTIVE_CPU_WORKERS 
    ))
    writer_proc.start()

    # --- 2. Start CPU Worker Pool ---
    cpu_pool = mp.Pool(
        processes=N_ACTIVE_CPU_WORKERS,
        maxtasksperchild=CPU_WORKER_BATCH_SIZE
    )
    cpu_procs = []
    for i in range(N_ACTIVE_CPU_WORKERS):
        p = cpu_pool.apply_async(cpu_worker, args=(cpu_work_queue, results_queue, i))
        cpu_procs.append(p)
    
    # --- 3. Start GPU Worker Pool ---
    gpu_pool = mp.Pool(
        processes=N_ACTIVE_GPU_WORKERS,
        maxtasksperchild=GPU_WORKER_BATCH_SIZE
    )
    gpu_procs = []
    for i in range(N_ACTIVE_GPU_WORKERS):
        p = gpu_pool.apply_async(gpu_worker, args=(
            file_queue, cpu_work_queue, i, TEMP_STAGE1_DIR
        ))
        gpu_procs.append(p)

    # --- 4. Fill Queues ---
    stage1_files_to_process = []
    stage2_files_to_process = []
    for f_audio in all_audio_files:
        track_id = f_audio.stem.replace(".low", "")
        if track_id in processed_track_ids:
            continue
        
        f_npz = TEMP_STAGE1_DIR / f"{track_id}.npz"
        if f_npz.exists():
            stage2_files_to_process.append(f_npz)
        else:
            stage1_files_to_process.append(f_audio)
            
    print(f"Populating CPU queue with {len(stage2_files_to_process)} existing intermediate files...")
    random.shuffle(stage2_files_to_process) 
    for f_npz in stage2_files_to_process:
        cpu_work_queue.put(f_npz)
        
    print(f"Populating GPU queue with {len(stage1_files_to_process)} new files...")
    random.shuffle(stage1_files_to_process) 
    for f_audio in stage1_files_to_process:
        file_queue.put(f_audio)
    
    files_processed_this_run = len(stage1_files_to_process) + len(stage2_files_to_process)
    if files_processed_this_run == 0 and len(processed_track_ids) > 0:
        print("All files are already processed. Shutting down workers.")
        for _ in range(N_ACTIVE_GPU_WORKERS):
            file_queue.put(None)
    elif files_processed_this_run == 0 and len(processed_track_ids) == 0:
         print("WARNING: No files to process and no completed files found. Check paths.")
         for _ in range(N_ACTIVE_GPU_WORKERS):
            file_queue.put(None)
    else:
        # --- 5. Add Sentinels ---
        for _ in range(N_ACTIVE_GPU_WORKERS):
            file_queue.put(None)
        
    # --- 6. Wait for all processes to finish ---
    for p in gpu_procs: p.get()
    gpu_pool.close()
    gpu_pool.join()
    
    for p in cpu_procs: p.get()
    cpu_pool.close()
    cpu_pool.join()
    
    writer_proc.join()
    
    # --- 7. Final Consolidation ---
    print("\n--- All processing complete. Consolidating final parquet file. ---")
    run_consolidation_stage() 

    end_time = time.time()
    total_time_seconds = end_time - start_time
    print("\n--- Phase 1 (Part A): Feature Extraction COMPLETE (maxtasksperchild Pipeline) ---")
    print(f"Total time: {total_time_seconds:.2f} seconds ({total_time_seconds/3600:.2f} hours)")
    
    if files_processed_this_run > 0 and total_time_seconds > 0:
        avg_time = total_time_seconds / files_processed_this_run
        print(f"Average throughput for this run: {1/avg_time:.2f} files/s")
        
    print(f"Output file saved to: {OUTPUT_FEATURES_FILE.resolve()}")
    print("You can now run 'build_jamendo_lookups.py' (Part B) to generate tags.")


if __name__ == "__main__":
    if sys.platform == "win32": 
        print("ERROR: This script must be run in WSL (or Linux).")
        sys.exit(1)
    
    try:
        mp.set_start_method('spawn', force=True) 
        print("Set multiprocessing start method to 'spawn' (required for CUDA).")
    except RuntimeError:
        pass # Already set
    
    main()
