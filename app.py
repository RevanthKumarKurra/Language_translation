from flask import Flask, jsonify, render_template, send_from_directory
import os
import uuid
import torch
import numpy as np
import torchaudio
import sounddevice as sd
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

app = Flask(__name__)
AUDIO_FOLDER = "static/audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def load_models():

    global speech_to_text_model
    global translation_model_tokenizer, translation_model
    global tel_text_speech_model, tel_text_speech_tokenizer
    global kan_text_speech_model, kan_text_speech_tokenizer
    global tam_text_speech_model, tam_text_speech_tokenizer
    global hin_text_speech_model, hin_text_speech_tokenizer

    print("Loading models...")

    speech_to_text_model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    translation_model_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    tel_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-tel")
    tel_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tel")

    kan_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-kan")
    kan_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kan")

    tam_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-tam")
    tam_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")

    hin_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
    hin_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

    print("All models loaded successfully.")



def text_translation(text,src_lang="tel_Telu",dest_lang="kan_Knda"):

    global translation_model_tokenizer, translation_model
    model = translation_model
    tokenizer = translation_model_tokenizer

    tokenizer.src_lang = src_lang 

    inputs = tokenizer(text, return_tensors="pt")

    input_text = tokenizer.decode(inputs.input_ids[0])

    translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(dest_lang), max_length=50
    )

    output_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return input_text,output_text


def text_to_speech(model,tokenizer,text):

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    
    output_array = output.detach().numpy()[0]
    sample_rate = model.config.sampling_rate

    return output_array,sample_rate


#print("model is loaded")

app = Flask(__name__)

def is_silent(data, threshold=0.01):
    return np.mean(np.abs(data)) < threshold

def record_voice(duration_silence=2, sample_rate=22000, chunk_duration=0.1):
    chunk_samples = int(sample_rate * chunk_duration)
    recorded_frames = []
    silent_chunks = 0
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')

    with stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk[:, 0]
            recorded_frames.append(chunk)

            if is_silent(chunk):
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks * chunk_duration >= duration_silence:
                break

    final_audio = np.concatenate(recorded_frames)
    return final_audio, sample_rate


#print("speak!!")

def speech_to_text(wave_array,sample_rate,src_lang="te"):

    global speech_to_text_model

    model = speech_to_text_model

   
    if wave_array.ndim == 1:
        wave_tensor = torch.tensor(wave_array).unsqueeze(0) 
    else:
        wave_tensor = torch.tensor(wave_array)               
        wave_tensor = torch.mean(wave_tensor, dim=0, keepdim=True)  


    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        wave_tensor = resampler(wave_tensor)


    transcription_ctc = model(wave_tensor, src_lang, "ctc")
    #print("CTC Transcription:", transcription_ctc)

    return transcription_ctc



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record/<person>')
def record(person):
    if person == "person1":
        speech_scr = "te"
        lang_src, lang_tgt = "tel_Telu", "kan_Knda"
        tts_model, tts_tokenizer = kan_text_speech_model, kan_text_speech_tokenizer
    else:
        speech_scr = "kn"
        lang_src, lang_tgt = "kan_Knda", "tam_Taml"
        tts_model, tts_tokenizer = tam_text_speech_model,tam_text_speech_tokenizer

    audio, sr = record_voice()
    print(audio)
    text = speech_to_text(audio, sr, src_lang=speech_scr)
    translated = text_translation(text, src_lang=lang_src, dest_lang=lang_tgt)
    print(translated)
    audio_data, audio_sr = text_to_speech(text = translated, model=tts_model, tokenizer=tts_tokenizer)

    filename = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(AUDIO_FOLDER, filename)
    torchaudio.save(path, torch.tensor(audio_data).unsqueeze(0), audio_sr)

    return jsonify({
        "text": text,
        "translated": translated,
        "audio_url": f"/static/audio/{filename}"
    })

@app.route('/delete_audio/<filename>')
def delete_audio(filename):
    try:
        os.remove(os.path.join(AUDIO_FOLDER, filename))
        return "Deleted", 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
