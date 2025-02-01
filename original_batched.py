import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from tqdm import tqdm
from vq.codec_encoder import CodecEncoder_Transformer
 
from vq.codec_decoder_vocos import CodecDecoderVocos
from argparse import ArgumentParser
from time import time
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch.nn as nn
from vq.module import SemanticEncoder
from torch.utils.data import Dataset, DataLoader
from typing import List
from collections import OrderedDict
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
import numpy as np

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
    
    return torch.tensor(padded_audios),torch.tensor(padded_feat_list), fname_list,audio_length

class WaveDataset(Dataset):
    def __init__(
        self,
        file_list,
        sampling_rate,
        audio_norm_scale: float = 1.0,
        root_dir: str = ""
    ):
        self.file_list = file_list
        self.sampling_rate = sampling_rate
        self.audio_norm_scale = audio_norm_scale
        self.hop_length = 320
        self.root_dir = root_dir
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def __getitem__(self, index):
        fname = self.file_list[index]
        fname = os.path.join(self.root_dir, fname)

        audio, sr = torchaudio.load(fname)
        if sr != self.sampling_rate:
            audio = Resample(sr, self.sampling_rate)(audio)
        if self.audio_norm_scale < 1.0:
            audio = audio * self.audio_norm_scale
 
        audio_pad = F.pad(audio, (160, 160))

        feat = self.feature_extractor(
                audio_pad,
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            ).data['input_features']

 
 
        return audio,feat, fname, int(audio.shape[1] / self.hop_length)
 
    def __len__(self):
        return len(self.file_list)

def save_vq_code(vq_codes: torch.Tensor, wav_paths: List[str], lengths: List[int], output_dir: str ):
    for i, wav_path in enumerate(wav_paths):
        relative_path = os.path.relpath(wav_path, args.input_dir)
        code_path = os.path.join(output_dir, 'vq_codes', relative_path.replace('.flac', '.npy'))
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        vq_code = vq_codes[i, 0,:lengths[i]]
        np.save(code_path, vq_code.detach().cpu().numpy().astype(np.int32))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=0, help='Local GPU device ID')
    parser.add_argument('--input-dir', type=str, default='/path/to/audio_folder', help='Input directory containing audio files')
    parser.add_argument('--flist_file', type=str, default='/path/to/file.txt', help='TSV file containing paths to audio files')
    parser.add_argument('--ckpt', type=str, default='/path/to/epoch=4-step=1400000.ckpt', help='Path to the model checkpoint')
    parser.add_argument('--output-dir', type=str, default='/path/to/saving_code_folder', help='Output directory for saving audio files')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for the DataLoader')
 
    device_id = int(os.getenv('LOCAL_RANK', 0))  
    args = parser.parse_args()
    sr = 16000

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


    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    semantic_model.to(device)
    SemanticEncoder_module.to(device)
    encoder.to(device)
    decoder.to(device)
    fc_post_a.to(device)
    fc_prior.to(device)

    #  assume 8 gpus on your devices,   flist_file can obtained using get_tsv.py
    df = pd.read_csv(args.flist_file, sep='\t', header=None, names=['filename', 'duration'], skiprows=1)
    file_list = df['filename'].tolist()
    # with open(args.flist_file, 'r') as f:
    #     file_list = [line.strip() for line in f if line.strip()]

    split_file_lists = np.array_split(file_list, 8) #8 gpus

 
    device_id = device_id
    current_file_list = split_file_lists[device_id]

    dataset = WaveDataset(file_list=current_file_list, sampling_rate=sr, root_dir=args.input_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_audio_batch
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
    print(f'End，time: {(et - st)/60:.2f} mins')