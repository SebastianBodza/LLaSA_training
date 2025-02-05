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

def pad_audio_batch(batch, longest_audio):
    """
    Expects each element in batch as a tuple (audio, feat, fname, audio_length, text).
    The longest_audio argument specifies the target length (in samples) for padding.
    """
    audio_list, feat_list, fname_list, audio_length, texts = zip(*batch)
    feat_list = list(feat_list)
    
    max_length_feat = max([feat.shape[1] for feat in feat_list])
    max_length = max_length_feat * 320  # hop_length = 320
    padded_audios = []
 
    for audio in audio_list:
        padding = longest_audio - audio.shape[1]
        if padding > 0:
            padded_audio = F.pad(audio, (0, padding), mode="constant", value=0)
        else:
            padded_audio = audio[:, :longest_audio]
        padded_audios.append(padded_audio)
    padded_audios = torch.stack(padded_audios)
    

    longest_feat = longest_audio // 320
    padded_feat_list = []
    for feat in feat_list:
        padding = longest_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode="constant", value=0)
        padded_feat_list.append(padded_feat)
    padded_feat_list = torch.stack(padded_feat_list)
    
    return padded_audios, padded_feat_list, fname_list, audio_length, texts
####################
# Dataset Function #
####################
def add_length(start, end):
    # Calculate lengths using start and end columns
    audio_len = np.array(end) - np.array(start)
    return {"audio_len": audio_len}

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
def save_tokenized_memmap(
    all_sequences: List[List[int]], output_dir: str, split: str,
    rank: int, batch_id: int, max_length: int
):
    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(
        output_dir,
        f"{split}_rank{rank}_partial{batch_id}_input_ids.memmap"
    )
    shape_path = os.path.join(
        output_dir,
        f"{split}_rank{rank}_partial{batch_id}_input_ids_shape.npy"
    )
    
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

# Utility for gathering objects across processes using a Gloo group.
def gather_results(result_list):
    """
    Gather a list of Python objects from all processes using a temporary Gloo process group.
    Returns a flattened list containing results from all processes.
    """
    world_size = dist.get_world_size()
    # Create a Gloo process group for gathering objects.
    gloo_group = dist.new_group(backend="gloo")
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, result_list, group=gloo_group)
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
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for the DataLoader')
    parser.add_argument('--max_length', type=int, default=4096, help='Max sequence length')
    args = parser.parse_args()

    # Initialize distributed backend and set device based on local rank.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    init_distributed()

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

    use_fp16 = False
    if use_fp16:
        semantic_model = semantic_model.half()
        SemanticEncoder_module = SemanticEncoder_module.half()
        encoder = encoder.half()
        decoder = decoder.half()
        fc_post_a = fc_post_a.half()
        fc_prior = fc_prior.half()


    # # Compile models using torch.compile for improved throughput.
    semantic_model = torch.compile(semantic_model, mode="reduce-overhead")
    SemanticEncoder_module = torch.compile(SemanticEncoder_module, mode="reduce-overhead")
    encoder = torch.compile(encoder, mode="reduce-overhead")
    decoder = torch.compile(decoder, mode="reduce-overhead")
    fc_post_a = torch.compile(fc_post_a, mode="reduce-overhead")
    fc_prior = torch.compile(fc_prior, mode="reduce-overhead")


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
    SPEECH_GEN_START_ID = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    SPEECH_GEN_END_ID = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    ######################
    # Process All Splits #
    ######################
    print(f"Loading dataset from {args.dataset_dir}")
    ds = load_from_disk(args.dataset_dir)
    for split in ds.keys():
        print(f"\nProcessing split: {split}")
        ds_split = ds[split]
        print(f"Sorting {split} split by audio length...")
        ds_split = ds_split.map(add_length, batched=True, num_proc=4, batch_size=5000, input_columns=["start", "end"],
                                cache_file_name=f"/media/bodza/Audio_Dataset/xcodec_raw_files/audio_len_{split}.arrow"
                                )
        # Getting the audio_len column and sorting it in memory
        audio_lens = ds_split["audio_len"]
        sorted_indices = np.argsort(audio_lens)
        # Reversed order
        sorted_indices = sorted_indices[::-1]
        ds_split = ds_split.select(sorted_indices,
                                   indices_cache_file_name=f"/media/bodza/Audio_Dataset/xcodec_raw_files/sorting_indices_{split}.json")
        longest_audio = len(ds_split[0]["audio"]["array"])

        # For testing take the last 32 samples
        # ds_split = ds_split.select(range(len(ds_split)-32, len(ds_split)))
        # For testing take the first 32 samples
        # ds_split = ds_split.select(range(64))
        dataset = WaveDataset(ds_split, target_sampling_rate=target_sr)
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=lambda batch: pad_audio_batch(batch, longest_audio=longest_audio),        
            )
        
    
        max_length = args.max_length
        all_final_sequences = [] 
        
        # Since we are not only GPU poor but also memory poor, we will save and clear the sequences periodically.
        save_interval = 5000  # 5000
        batch_counter = 0
        partial_counter = 0
        
        print("Processing batches ...")
        st = time()
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
            for batch in tqdm(dataloader, desc=f"Processing split {split}"):
                batch_counter += 1
                model_start_time = time()
                wavs, feats, wav_paths, lengths, texts = batch
                wavs = wavs.to(device)
            
                # 1) Codec encoder to get speech representation
                vq_emb = encoder(wavs)  # [batch, time//down, 1024]
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
                # Expected vq_code shape: [batch, 1, frames]
                batch_size = vq_code.size(0)
                vq_code = vq_code.to(device='cpu') 

                text_inputs = [f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>" for text in texts]
                batched_text_ids = tokenizer.batch_encode_plus(text_inputs, 
                                                               add_special_tokens=False, 
                                                               padding=False, 
                                                               truncation=False
                                                               )["input_ids"]
                for i in range(batch_size):
                    text_ids = batched_text_ids[i]
                    speech_codes_tensor = vq_code[i, 0, : lengths[i]]
                    speech_codes = speech_codes_tensor.tolist()
                    speech_token_strs = [f"<|s_{int(code)}|>" for code in speech_codes]
                    speech_ids = (
                        [SPEECH_GEN_START_ID]
                        + [tokenizer.convert_tokens_to_ids(token) for token in speech_token_strs]
                        + [SPEECH_GEN_END_ID]
                    )
                    
                    MAX_TEXT_SPACE = max_length - len(speech_ids)
                    if MAX_TEXT_SPACE < 0:
                        continue
                    truncated_text = text_ids[:MAX_TEXT_SPACE]
                    final_sequence = (truncated_text + speech_ids +
                                    [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids)))
                    final_sequence = final_sequence[:max_length]
                    all_final_sequences.append(final_sequence)
                
                # Save and clear periodically to release memory.
                if batch_counter % save_interval == 0:
                    save_tokenized_memmap(all_final_sequences, args.output_dir, split, local_rank, partial_counter, max_length )
                    all_final_sequences.clear()  
                    partial_counter += 1

            if all_final_sequences:
                save_tokenized_memmap(
                    all_final_sequences,
                    args.output_dir,
                    split,
                    local_rank,
                    partial_counter,
                    max_length
                )
                all_final_sequences.clear()
        et = time()
        print(f"Processing split '{split}' completed in {(et - st) / 60:.2f} mins")
    # HF_HOME="/media/bodza/Audio_Dataset/hf_cache/" torchrun --nproc_per_node=2 original_translate_to_dataset_o3.py --dataset-dir="/media/bodza/Audio_Dataset/podcast_dataset/" --ckpt="/media/bodza/Audio_Dataset/xcodec_raw_files/xcodec2/ckpt/epoch=4-step=1400000.ckpt" --output-dir="/media/bodza/Audio_Dataset/xcodec_raw_files/output_original"
