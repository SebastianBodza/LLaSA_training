import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from vq.codec_encoder import CodecEncoder_Transformer
 
from vq.codec_decoder_vocos import CodecDecoderVocos

from argparse import ArgumentParser
from time import time
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch.nn as nn
from vq.module import SemanticEncoder
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from typing import List
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd

from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk

def pad_audio_batch(batch):
    audio_list, feat_list, fname_list, audio_length = zip(*batch)
    feat_list = list(feat_list)
    
    max_length_feat = max([feat.shape[1] for feat in feat_list])
    max_length =  max_length_feat *320
    padded_audios = []
 
    for audio in audio_list:
        padding = max_length - audio.shape[1] 
        if padding > 0:
 
            padded_audio = F.pad(audio,   (0, padding) , mode='constant', value=0) 
        else:
            padded_audio = audio[:,:max_length]
        padded_audios.append(padded_audio)
    padded_audios = torch.stack(padded_audios)
    padded_feat_list = []
    for feat in feat_list:
        padding = max_length_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode='constant', value=0)
        padded_feat_list.append(padded_feat)
 
 
    padded_feat_list = torch.stack(padded_feat_list)
    
    return padded_audios, padded_feat_list, fname_list, audio_length

class WaveDataset(Dataset):
    def __init__(
        self,
        ds,
        target_sampling_rate: int,
        audio_norm_scale: float = 1.0
    ):
        self.ds = ds
        self.target_sampling_rate = target_sampling_rate
        self.audio_norm_scale = audio_norm_scale
        self.hop_length = 320
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


    def __getitem__(self, index):
        record = self.ds[index]
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
        return audio, feat, fname, audio_length

    def __len__(self):
        return len(self.ds)

def save_vq_code(vq_codes: torch.Tensor, wav_paths: List[str], lengths: List[int], output_dir: str ):
    for i, wav_path in enumerate(wav_paths):
        code_path = os.path.join(
            output_dir, "vq_codes", os.path.basename(wav_path).replace(".flac", ".npy")
        )
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        vq_code = vq_codes[i, 0,:lengths[i]]
        np.save(code_path, vq_code.detach().cpu().numpy().astype(np.int32))


def init_distributed():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=0, help='Local GPU device ID')
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/path/to/dataset_folder",
        help="Directory containing the disk-saved dataset (loaded via load_from_disk)",
    )
    parser.add_argument('--ckpt', type=str, default='/path/to/epoch=4-step=1400000.ckpt', help='Path to the model checkpoint')
    parser.add_argument('--output-dir', type=str, default='/path/to/saving_code_folder', help='Output directory for saving audio files')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for the DataLoader')
    args = parser.parse_args()

    # Initialize distributed backend
    init_distributed()
    local_rank = args.local_rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    target_sr = 16000

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'loading codec checkpoint from {args.ckpt}')
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

    print(f"Loading dataset from {args.dataset_dir}")
    ds = load_from_disk(args.dataset_dir)
    ds_split = ds["test"] if "test" in ds.keys() else ds
    ds_split = ds_split.select(list(range(32)))


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

    st = time()
    for batch in tqdm(dataloader, desc="processing"):
        wavs,feats,wav_paths, lengths = batch
        wavs = wavs.to(device)


        with torch.no_grad():
 
            vq_emb = encoder(wavs )
            vq_emb = vq_emb.transpose(1, 2)

 
            semantic_target = semantic_model(feats[:,0,:,:].to(device))
            semantic_target = semantic_target.hidden_states[16]
            semantic_target = semantic_target.transpose(1, 2)
            semantic_target = SemanticEncoder_module(semantic_target)

            vq_emb = torch.cat([semantic_target, vq_emb], dim=1)
            vq_emb = fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

            _, vq_code, _ = decoder(vq_emb, vq=True)

        save_vq_code(vq_code, wav_paths, lengths, args.output_dir)

    et = time()
    if local_rank == 0:
        print(f"End, total time: {(et - st) / 60:.2f} mins")
