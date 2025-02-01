import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from vq.codec_encoder import CodecEncoder_Transformer
from vq.codec_decoder_vocos import CodecDecoderVocos
from argparse import ArgumentParser
from time import time
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, AutoTokenizer
import torch.nn as nn
from vq.module import SemanticEncoder
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from typing import List
from torchaudio.transforms import Resample
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
import torch.distributed as dist

#####################
# Utility Functions #
#####################

def pad_audio_batch(batch):
    audio_list, feat_list, fname_list, audio_length, texts = zip(*batch)
    feat_list = list(feat_list)
    
    max_length_feat = max([feat.shape[1] for feat in feat_list])
    max_length = max_length_feat * 320  # hop_length = 320
    padded_audios = []
 
    for audio in audio_list:
        padding = max_length - audio.shape[1]
        if padding > 0:
            padded_audio = F.pad(audio, (0, padding), mode="constant", value=0)
        else:
            padded_audio = audio[:, :max_length]
        padded_audios.append(padded_audio)
    padded_audios = torch.stack(padded_audios)
    
    padded_feat_list = []
    for feat in feat_list:
        padding = max_length_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode="constant", value=0)
        padded_feat_list.append(padded_feat)
    padded_feat_list = torch.stack(padded_feat_list)
    
    return padded_audios, padded_feat_list, fname_list, audio_length, texts

####################
# Dataset Function #
####################

class WaveDataset(Dataset):
    """
    Expects the disk-saved dataset to have an 'audio' column (with keys 'array',
    'sampling_rate', and optionally 'path') AND a 'text' column.
    """
    def __init__(self, ds, target_sampling_rate: int, audio_norm_scale: float = 1.0):
        self.ds = ds
        self.target_sampling_rate = target_sampling_rate
        self.audio_norm_scale = audio_norm_scale
        self.hop_length = 320
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def __getitem__(self, index):
        record = self.ds[index]
        # Process audio
        audio_np = record["audio"]["array"]
        sr = record["audio"]["sampling_rate"]
        audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        if sr != self.target_sampling_rate:
            audio = Resample(sr, self.target_sampling_rate)(audio)
        if self.audio_norm_scale < 1.0:
            audio = audio * self.audio_norm_scale
        audio_pad = F.pad(audio, (160, 160))
        feat = self.feature_extractor(
            audio_pad,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt"
        ).data["input_features"]
        audio_length = int(audio.shape[1] / self.hop_length)
        fname = record["audio"].get("path", f"sample_{index}")
        
        # Also retrieve text (assumes a column 'text' exists)
        text = record.get("text", "")
        return audio, feat, fname, audio_length, text

    def __len__(self):
        return len(self.ds)

###########################
# Saving Tokenized Output #
###########################

def save_tokenized_memmap(all_sequences: List[List[int]], output_dir: str, 
                          split: str, max_length: int):
    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
    shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")
    
    all_sequences = np.array(all_sequences, dtype=np.int32)
    num_sequences = all_sequences.shape[0]
    
    arr = np.memmap(
        memmap_path, dtype=np.int32, mode="w+", shape=(num_sequences, max_length)
    )
    arr[:] = all_sequences
    arr.flush()
    np.save(shape_path, np.array([num_sequences, max_length]))
    print(f"Saved {num_sequences} sequences of length {max_length} to {memmap_path}")
    print(f"Shape saved in {shape_path}")

###################
# Distributed Init#
###################

def init_distributed():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")

# Utility for gathering objects across processes.
def gather_results(result_list):
    """
    Gather a list of Python objects from all processes using a temporary Gloo process group.
    Returns a flattened list containing results from all processes.
    """
    world_size = dist.get_world_size()
    # Create a new process group using Gloo (on CPU)
    gloo_group = dist.new_group(backend="gloo")
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, result_list, group=gloo_group)
    # Flatten the list:
    combined = []
    for sublist in all_results:
        combined.extend(sublist)
    return combined




####################
# Main Processing  #
####################

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=0, help='Local GPU device ID')
    parser.add_argument("--dataset-dir", type=str, default="/path/to/dataset_folder",
                        help="Directory containing the disk-saved dataset (loaded via load_from_disk)")
    parser.add_argument('--ckpt', type=str, default='/path/to/epoch=4-step=1400000.ckpt',
                        help='Path to the model checkpoint')
    parser.add_argument('--output-dir', type=str, default='/path/to/saving_code_folder',
                        help='Output directory for saving tokenized sequences')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for the DataLoader')
    # Processing parameters:
    parser.add_argument('--max_length', type=int, default=4096, help='Max sequence length')
    args = parser.parse_args()
    # Initialize distributed backend
    init_distributed()
    local_rank = args.local_rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


    target_sr = 16000
    os.makedirs(args.output_dir, exist_ok=True)
    
    ############################
    # Load Checkpoint & Models #
    ############################
    print(f'Loading codec checkpoint from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt = ckpt['state_dict']

    filtered_state_dict_codec = OrderedDict()
    filtered_state_dict_semantic_encoder = OrderedDict()
    filtered_state_dict_gen = OrderedDict()
    filtered_state_dict_fc_post_a = OrderedDict()
    filtered_state_dict_fc_prior = OrderedDict()

    for key, value in ckpt.items():
        if key.startswith('CodecEnc.'):
            new_key = key[len('CodecEnc.'):]
            filtered_state_dict_codec[new_key] = value
        elif key.startswith('generator.'):
            new_key = key[len('generator.'):]
            filtered_state_dict_gen[new_key] = value
        elif key.startswith('fc_post_a.'):
            new_key = key[len('fc_post_a.'):]
            filtered_state_dict_fc_post_a[new_key] = value
        elif key.startswith('SemanticEncoder_module.'):
            new_key = key[len('SemanticEncoder_module.'):]
            filtered_state_dict_semantic_encoder[new_key] = value
        elif key.startswith('fc_prior.'):
            new_key = key[len('fc_prior.'):]
            filtered_state_dict_fc_prior[new_key] = value

    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)
    semantic_model.eval()

    SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
    SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)
    SemanticEncoder_module.eval()

    encoder = CodecEncoder_Transformer()
    encoder.load_state_dict(filtered_state_dict_codec)
    encoder.eval()

    decoder = CodecDecoderVocos()
    decoder.load_state_dict(filtered_state_dict_gen)
    decoder.eval()

    fc_post_a = nn.Linear(2048, 1024)
    fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
    fc_post_a.eval()

    fc_prior = nn.Linear(2048, 2048)
    fc_prior.load_state_dict(filtered_state_dict_fc_prior)
    fc_prior.eval()

    semantic_model.to(device)
    SemanticEncoder_module.to(device)
    encoder.to(device)
    decoder.to(device)
    fc_post_a.to(device)
    fc_prior.to(device)

    ############################
    # Load Tokenizer & Add Tokens
    ############################
    # Adjust the pretrained tokenizer as necessary.
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    # Define the special tokens needed.
    extra_tokens = [
        "<|TEXT_GENERATION_START|>",
        "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>",
        "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>",
        "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>",
        "<|SPEECH_UNDERSTANDING_END|>",
    ]
    # Add a large range of speech tokens.
    new_speech_tokens = [f"<|s_{i}|>" for i in range(65536)]
    all_new_tokens = extra_tokens + new_speech_tokens
    tokenizer.add_tokens(all_new_tokens)
    # Set the pad token id (adjust according to your tokenizer/model)
    tokenizer.pad_token_id = 128001

    #########################
    # Load & Subset Dataset #
    #########################
    print(f"Loading dataset from {args.dataset_dir}")
    ds = load_from_disk(args.dataset_dir)
    # Prefer the "test" split if it exists; otherwise use the entire dataset.
    for split in ds.keys():
        print(f"\nProcessing split: {split}")
        ds_split = ds[split]


        dataset = WaveDataset(ds_split, target_sampling_rate=target_sr)
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=pad_audio_batch,
        )

        #####################
        # Processing & Save #
        #####################
        max_length = args.max_length
        all_final_sequences = []  # List to accumulate tokenized sequences

        print("Processing batches ...")
        st = time()
        for batch in tqdm(dataloader, desc="processing"):
            wavs, feats, wav_paths, lengths, texts = batch
            wavs = wavs.to(device)
            
            with torch.no_grad():
                # 1) Codec encoder to get speech representation
                vq_emb = encoder(wavs)  # e.g., [batch, time//down, 1024]
                vq_emb = vq_emb.transpose(1, 2)  # [batch, 1024, frames]

                # 2) Semantic processing
                semantic_target = semantic_model(feats[:, 0, :, :].to(device))
                semantic_target = semantic_target.hidden_states[16]
                semantic_target = semantic_target.transpose(1, 2)
                semantic_target = SemanticEncoder_module(semantic_target)

                # 3) Concatenate and process with fc_prior
                vq_emb = torch.cat([semantic_target, vq_emb], dim=1)
                vq_emb = fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

                # 4) Pass through decoder quantization part to get final speech tokens
                _, vq_code, _ = decoder(vq_emb, vq=True)
                # vq_code shape: [batch, frames]

            # For each sample in the batch, convert the speech codes into token IDs
            # and combine these with the tokenized text.
            batch_size = vq_code.size(0)
            for i in range(batch_size):
                # Process text: add understanding start/end markers
                text = texts[i]
                text_input = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
                text_ids = tokenizer.encode(text_input, add_special_tokens=False)
                
                # Process speech: convert each code into a token.
                # Note: You can adjust the conversion as needed.
                speech_codes = vq_code[i, 0, : lengths[i]]   # Now speech_codes is 1D
                speech_ids = ([tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
                            + [tokenizer.convert_tokens_to_ids(f"<|s_{int(code.item())}|>")
                                for code in speech_codes.cpu()]
                            + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")])
                
                # Determine space available for text tokens.
                MAX_TEXT_SPACE = max_length - len(speech_ids)
                if MAX_TEXT_SPACE < 0:
                    # Skip samples that produce too many speech codes.
                    continue
                truncated_text = text_ids[:MAX_TEXT_SPACE]
                
                final_sequence = (truncated_text + speech_ids +
                                [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids)))
                final_sequence = final_sequence[:max_length]
                
                all_final_sequences.append(final_sequence)
        et = time()
        print(f"Processing split '{split}' completed in {(et - st) / 60:.2f} mins")
    
        # Gather sequences from all processes using the Gloo group
        if dist.is_initialized():
            all_final_sequences = gather_results(all_final_sequences)
    
        # Only rank 0 writes the final memmap for this split.
        if local_rank == 0:
            print(f"Saving tokenized sequences for split '{split}' to memmap...")
            save_tokenized_memmap(all_final_sequences, args.output_dir, split, max_length)

