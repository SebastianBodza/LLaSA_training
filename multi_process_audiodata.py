import os
import numpy as np
from datasets import load_from_disk
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

def setup_tokenizer():
    # RIP Llamapoor europeans -> ungated unsloth version 
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    
    Start_End_tokens = [
        "<|TEXT_GENERATION_START|>",
        "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>",
        "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>",
        "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>",
        "<|SPEECH_UNDERSTANDING_END|>",
    ]

    new_speech_tokens = [f"<|s_{i}|>" for i in range(65536)]
    all_new_tokens = Start_End_tokens + new_speech_tokens
    tokenizer.add_tokens(all_new_tokens)
    # the llasa uses the end_of_text_token as the pad_token_id and not the eot_token!
    tokenizer.pad_token_id = 128001
    return tokenizer

def process_gpu_chunk(rank, world_size, dataset_split, output_dir, tokenizer, max_length, debug=False):
    from xcodec_processing import process_batch
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    # Calculate chunk size for this GPU
    total_samples = len(dataset_split)
    chunk_size = total_samples // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != (world_size - 1) else total_samples
    
    # Get this GPU's chunk of data
    if debug:
        chunk = dataset_split.select(range(start_idx, min(start_idx + 10, end_idx)))
    else:
        chunk = dataset_split.select(range(start_idx, end_idx))
    
    print(f"GPU {rank} processing samples {start_idx} to {end_idx}")
    
    # Process the chunk
    processed_chunk = chunk.map(
        process_batch,
        batched=True,
        batch_size=32,
        remove_columns=chunk.column_names,
        fn_kwargs={
            "rank": rank,
            "tokenizer": tokenizer
        }
    )
    
    # Filter valid sequences
    valid_sequences = [seq for seq in processed_chunk["input_ids"] if seq is not None]
    
    if len(valid_sequences) > 0:
        # Save this GPU's results to a temporary file
        temp_memmap_path = os.path.join(output_dir, f"temp_{rank}.memmap")
        arr = np.memmap(
            temp_memmap_path,
            dtype=np.int32,
            mode="w+",
            shape=(len(valid_sequences), max_length)
        )
        arr[:] = np.array(valid_sequences, dtype=np.int32)
        arr.flush()
        
    return len(valid_sequences)

def main(dataset_name, output_dir, max_length, debug=False):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    print("Loading dataset...")
    dataset = load_from_disk(dataset_name)
    tokenizer = setup_tokenizer()
    
    # Get available splits
    splits = dataset.keys()
    print(f"Found splits: {splits}")
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    for split in splits:
        print(f"\nProcessing split: {split}")
        split_data = dataset[split]

        split_data = split_data.select(range(int(len(split_data)*0.1)))
        # TODO: ATTENTION loading only a subset

        # Start multiple processes for GPU processing
        mp.spawn(
            process_gpu_chunk,
            args=(world_size, split_data, output_dir, tokenizer, max_length, debug),
            nprocs=world_size,
            join=True
        )
        
        # Combine results from all GPUs
        all_sequences = []
        for rank in range(world_size):
            temp_path = os.path.join(output_dir, f"temp_{rank}.memmap")
            if os.path.exists(temp_path):
                temp_arr = np.memmap(temp_path, dtype=np.int32, mode="r")
                all_sequences.extend(temp_arr.reshape(-1, max_length))
                os.remove(temp_path)  # Clean up temporary file
        
        # Save final combined results
        if len(all_sequences) > 0:
            final_memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
            shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")
            
            arr = np.memmap(
                final_memmap_path,
                dtype=np.int32,
                mode="w+",
                shape=(len(all_sequences), max_length)
            )
            arr[:] = np.array(all_sequences, dtype=np.int32)
            arr.flush()
            np.save(shape_path, np.array([len(all_sequences), max_length]))
            
            print(f"\n=== {split} Split Summary ===")
            print(f"Saved {len(all_sequences)} sequences of length {max_length}")
            print(f"Memmap file size: {os.path.getsize(final_memmap_path)/1e6:.2f}MB")
            print(f"Shape: {np.load(shape_path)}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    dataset_name = "/"
    output_dir = "/"
    max_length = 4096
    debug = False

    main(dataset_name, output_dir, max_length, debug)