import os
import numpy as np
from datasets import load_dataset, load_from_disk
import torch
import torchaudio
from transformers import AutoTokenizer
from xcodec2 import XCodec2Model  # Assuming xcodec2 is properly installed

def preprocess_dataset(
    dataset_name: str,
    split: str,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    xcodec2_model_name: str = "HKUST-Audio/xcodec2",
    sample_rate: int = 16000,
    max_length: int = 2048
):
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split=split)
        # if the dataset is a dataset from disk load it from disk
    except:
        dataset = load_from_disk(dataset_name)
    
    # Initialize tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    Start_End_tokens = [
        '<|TEXT_GENERATION_START|>',
        '<|TEXT_GENERATION_END|>',
        '<|TEXT_UNDERSTANDING_START|>',
        '<|TEXT_UNDERSTANDING_END|>',
        '<|SPEECH_GENERATION_START|>',
        '<|SPEECH_GENERATION_END|>',
        '<|SPEECH_UNDERSTANDING_START|>',
        '<|SPEECH_UNDERSTANDING_END|>'
    ]


    new_speech_tokens = [f'<|s_{i}|>' for i in range(65536)]   
    all_new_tokens = Start_End_tokens + new_speech_tokens
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    print(f"Added {num_added_tokens} speech tokens to the tokenizer.")

    tokenizer.pad_token = tokenizer.eos_token

    # Load audio codec model
    codec_model = XCodec2Model.from_pretrained(xcodec2_model_name).eval().cuda()

    # Create memory-mapped arrays
    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
    shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")

    # Process all examples
    all_sequences = []
    for example in dataset:
        # Process text
        text = f"<|TEXT_UNDERSTANDING_START|>{example['text']}<|TEXT_UNDERSTANDING_END|>"
        text_ids = tokenizer.encode(text, add_special_tokens=False)

        # Process audio
        waveform = torch.tensor(example['audio']['array']).float()
        if example['audio']['sampling_rate'] != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, 
                example['audio']['sampling_rate'], 
                sample_rate
            )
        with torch.no_grad():
            speech_codes = codec_model.encode_code(waveform.unsqueeze(0).cuda())[0, 0]  # Get first codebook
        speech_ids = [
            tokenizer.convert_tokens_to_ids(f"<|s_{code}|>")
            for code in speech_codes.cpu().numpy()
        ]
        speech_ids = [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")] + speech_ids
        speech_ids.append(tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>"))

        # Combine and truncate/pad
        combined = text_ids + speech_ids
        combined = combined[:max_length]
        padded = combined + [tokenizer.pad_token_id] * (max_length - len(combined))
        all_sequences.append(padded)

    # Save to memmap
    arr = np.memmap(memmap_path, dtype=np.int32, mode="w+", shape=(len(all_sequences), max_length))
    arr[:] = np.array(all_sequences, dtype=np.int32)
    arr.flush()
    
    # Save shape
    np.save(shape_path, np.array([len(all_sequences), max_length]))

if __name__ == "__main__":
    preprocess_dataset(
        dataset_name="your_dataset_name",
        split="train",
        output_dir="./processed_data",
        sample_rate=16000,
        max_length=2048
    )