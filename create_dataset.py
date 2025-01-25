import os
import numpy as np
from datasets import load_dataset, load_from_disk
import torch
import torchaudio
from transformers import AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model
from tqdm import tqdm

def preprocess_dataset(
    dataset_name: str,
    split: str,
    output_dir: str,
    tokenizer_name: str = "unsloth/Llama-3.2-1B-Instruct",
    xcodec2_model_name: str = "HKUST-Audio/xcodec2",
    sample_rate: int = 16000,
    max_length: int = 2048,
    debug: bool = False  # New debug flag
):
    # Load dataset
    print("Loading dataset...")
    # try:
    #     dataset = load_dataset(dataset_name, split=split)
    # except:
    #     dataset = load_from_disk(dataset_name)
    dataset = load_from_disk(dataset_name)

    if debug:
        print("\n*** DEBUG MODE ACTIVATED - PROCESSING 10 SAMPLES ***\n")
        dataset = dataset["train"].select(range(10))  # Get first 10 samples

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if debug:
        print("Loaded tokenizer:", tokenizer)
        print("Original vocabulary size:", len(tokenizer))

    # Add special tokens
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
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    # Use the 128001 token id as the pad token 
    tokenizer.pad_token_id = 128001
    
    print(f"\nAdded {num_added_tokens} special tokens")
    print("New vocabulary size:", len(tokenizer))
    print("Pad token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)

    # Load codec model
    codec_model = XCodec2Model.from_pretrained(xcodec2_model_name).eval().cuda()
    if debug:
        print("\nLoaded XCodec2 model:", codec_model.__class__.__name__)

    # Prepare memmap
    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
    shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")

    all_sequences = []
    for idx, example in tqdm(enumerate(dataset)):
        # Process text
        text = f"<|TEXT_UNDERSTANDING_START|>{example['text']}<|TEXT_UNDERSTANDING_END|>"
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Process audio
        waveform = torch.tensor(example["audio"]["array"]).float()

        if example["audio"]["sampling_rate"] != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, example["audio"]["sampling_rate"], sample_rate
            )

        with torch.no_grad():
            speech_codes = codec_model.encode_code(waveform.unsqueeze(0).cuda())[0, 0]

        speech_ids = (
            [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
            + [tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes.cpu().numpy()]
            + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
        )


        # Calculate available space
        MAX_TEXT_SPACE = max_length - len(speech_ids)
        if MAX_TEXT_SPACE < 0:
            raise ValueError(f"Speech sequence too long ({len(speech_ids)} tokens) for max_length {max_length}")

        # Truncate text to fit
        truncated_text = text_ids[:MAX_TEXT_SPACE]
        
        if debug:
            print(f"\nTruncated text tokens: {len(truncated_text)} (max available: {MAX_TEXT_SPACE})")

        # Build final sequence
        final_sequence = (
            truncated_text
            + speech_ids
            + [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids))
        )[:max_length]

        all_sequences.append(final_sequence)

    # Save to disk
    arr = np.memmap(memmap_path, dtype=np.int32, mode="w+", shape=(len(all_sequences), max_length))
    arr[:] = np.array(all_sequences, dtype=np.int32)
    arr.flush()
    np.save(shape_path, np.array([len(all_sequences), max_length]))

    print("\n=== Debug Summary ===")
    print(f"Saved {len(all_sequences)} sequences of length {max_length}")
    print(f"Memmap file size: {os.path.getsize(memmap_path)/1e6:.2f}MB")
    print(f"Example shape: {np.load(shape_path)}")


if __name__ == "__main__":
    preprocess_dataset(
        dataset_name="/media/bodza/Audio_Dataset/podcast_dataset/",
        split="train",
        output_dir="/media/bodza/Audio_Dataset/xcodec2_dataset/",
        sample_rate=16000,
        max_length=4096,
        debug=False  # Activate debug mode
    )