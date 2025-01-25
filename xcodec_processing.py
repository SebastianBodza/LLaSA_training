import torch
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio

codec_model = None


def process_batch(batch, rank=None, audio_column_name="audio", batch_size=32, tokenizer=None):
    global codec_model
    max_length = 4096 

    print("======== [ Rank: ", rank, "] ========")


    try:
        # Probably not necessary anymore 
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass 

    if codec_model is None:
        print("Loading XCodec2 model...")
        device = f"cuda:{rank}" if rank is not None else "cuda"
        codec_model = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2")
        print(f"Moving model to {device}")
        codec_model.to(device)
        codec_model.semantic_model.to(device)
    

    if isinstance(batch[audio_column_name], list):
        all_sequences = []
        for audio, text in zip(batch[audio_column_name], batch["text"]):
            text_input = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
            text_ids = tokenizer.encode(text_input, add_special_tokens=False)
            
            waveform = torch.tensor(audio["array"]).float()
            if audio["sampling_rate"] != 16000:
                waveform = torchaudio.functional.resample(
                    waveform, audio["sampling_rate"], 16000
                )
            

            with torch.no_grad():
                speech_codes = codec_model.encode_code(waveform.unsqueeze(0))[0, 0]

            speech_ids = (
                [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
                + [tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes.cpu().numpy()]
                + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
            )

            MAX_TEXT_SPACE = max_length - len(speech_ids)
            if MAX_TEXT_SPACE < 0:
                continue

            truncated_text = text_ids[:MAX_TEXT_SPACE]
            
            final_sequence = (
                truncated_text
                + speech_ids
                + [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids))
            )[:max_length]

            all_sequences.append(final_sequence)
        
        batch["input_ids"] = all_sequences

    else:
        # Single sample processing
        text_input = f"<|TEXT_UNDERSTANDING_START|>{batch['text']}<|TEXT_UNDERSTANDING_END|>"
        text_ids = tokenizer.encode(text_input, add_special_tokens=False)
        
        waveform = torch.tensor(batch[audio_column_name]["array"]).float()
        if batch[audio_column_name]["sampling_rate"] != 16000:
            waveform = torchaudio.functional.resample(
                waveform, batch[audio_column_name]["sampling_rate"], 16000
            )
        
        waveform = waveform.to(device)

        with torch.no_grad():
            speech_codes = codec_model.encode_code(waveform.unsqueeze(0))[0, 0]

        speech_ids = (
            [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
            + [tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes.cpu().numpy()]
            + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
        )

        MAX_TEXT_SPACE = max_length - len(speech_ids)
        if MAX_TEXT_SPACE < 0:
            batch["input_ids"] = None
            return batch

        truncated_text = text_ids[:MAX_TEXT_SPACE]
        
        final_sequence = (
            truncated_text
            + speech_ids
            + [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids))
        )[:max_length]

        batch["input_ids"] = final_sequence
    
    return batch