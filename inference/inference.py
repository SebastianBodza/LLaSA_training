from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
import time

print("Starting the TTS process...")

llasa_3b ='/media/bodza/Audio_Dataset/Llasa-Kartoffel-1B-v0.2'
print(f"Loading models from: {llasa_3b}")

print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(llasa_3b)
model.eval() 
model.to('cuda')
print("LLaSA model loaded and moved to CUDA")

from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUST-Audio/xcodec2"  
print(f"Loading XCodec2 model from: {model_path}")

Codec_model = XCodec2Model.from_pretrained(model_path)
Codec_model.eval().cuda()   
print("XCodec2 model loaded and moved to CUDA")

input_text = 'DeepSeek ist ein chinesisches KI-Startup, das sich auf die Entwicklung fortschrittlicher Sprachmodelle und künstlicher Intelligenz spezialisiert hat. Das Unternehmen gewann internationale Aufmerksamkeit mit der Veröffentlichung seines im Januar 2025 vorgestellten Modells DeepSeek R1, das mit etablierten KI-Systemen wie ChatGPT von OpenAI und Claude von Anthropic konkurriert.'
print(f"\nInput text length: {len(input_text)} characters")
print(f"Input text: {input_text[:100]}...")

def ids_to_speech_tokens(speech_ids):
    print(f"Converting {len(speech_ids)} IDs to speech tokens")
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    print(f"Extracting speech IDs from {len(speech_tokens_str)} tokens")
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Warning: Unexpected token: {token_str}")
    return speech_ids

print("\nStarting text-to-speech generation...")
start_time = time.time()

with torch.no_grad():
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
    print("Text formatted with understanding tags")

    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
    ]
    print("Chat template prepared")

    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    input_ids = input_ids.to('cuda')
    print(f"Input tokenized, shape: {input_ids.shape}")

    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    print("\nGenerating speech tokens...")
    
    outputs = model.generate(
        input_ids,
        max_length=2048,
        eos_token_id=speech_end_id,
        do_sample=True,    
        top_p=1,
        temperature=0.8,
    )
    print(f"Generation complete. Output shape: {outputs.shape}")

    generated_ids = outputs[0][input_ids.shape[1]:-1]
    print(f"Generated IDs shape: {generated_ids.shape}")

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
    print(f"Decoded to {len(speech_tokens)} speech tokens")

    speech_tokens = extract_speech_ids(speech_tokens)
    print(f"Extracted {len(speech_tokens)} speech IDs")

    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
    print(f"Speech tokens tensor shape: {speech_tokens.shape}")

    print("\nDecoding speech tokens to waveform...")
    gen_wav = Codec_model.decode_code(speech_tokens) 
    print(f"Generated waveform shape: {gen_wav.shape}")

print(f"\nSaving audio to gen.wav")
sf.write("gen.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)

end_time = time.time()
print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
print("Process completed successfully!")