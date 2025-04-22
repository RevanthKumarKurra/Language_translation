from flask import Flask, jsonify, render_template,request
from flask_cors import CORS
import sounddevice as sd
import numpy as np
import time
import torch
import librosa
from transformers import AutoModel,AutoModelForSeq2SeqLM,AutoTokenizer,VitsModel, AutoTokenizer
import torchaudio
import os
import uuid
from tensorflow.keras.saving import load_model
import scipy.io.wavfile as wav
from custom_transformers_model import Translator,Transformer
from tokenizers import Tokenizer

app = Flask(__name__)
CORS(app)

AUDIO_FOLDER = "static/audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def load_models():

    global speech_to_text_model
    global tokenizer, translation_model
    global tel_text_speech_model, tel_text_speech_tokenizer
    global kan_text_speech_model, kan_text_speech_tokenizer
    global tam_text_speech_model, tam_text_speech_tokenizer
    global hin_text_speech_model, hin_text_speech_tokenizer
    global key_dict

    print("Loading models...")

    speech_to_text_model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    tokenizer = Tokenizer.from_file("./tokenizer")
    translation_model = load_model("./lang_transformers.keras")

    tel_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-tel")
    tel_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tel")

    kan_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-kan")
    kan_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kan")

    tam_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-tam")
    tam_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")

    hin_text_speech_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
    hin_text_speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

    key_dict = {
        "hindi":["hi","<hin_start>","<hin_end>",hin_text_speech_model,hin_text_speech_tokenizer],
        "kannada":["kn","<kan_start>","<kan_end>",kan_text_speech_model,kan_text_speech_tokenizer],
        "tamil":["ta","<tam_start>","<tam_end>",tam_text_speech_model,tam_text_speech_tokenizer],
        "telugu":["te","<tel_start>","<tel_end>",tel_text_speech_model,tel_text_speech_tokenizer] }

    print("All models loaded successfully.")

def text_translation(text,src_srt=None,src_end=None,dest_srt=None,dest_end=None,):

    global tokenizer, translation_model
    model = translation_model
    tokenizer = tokenizer

    #tokenizer.src_lang = src_lang 

    inputs = src_srt + " " + text +" "+src_end

    #input_text = tokenizer.decode(inputs.input_ids[0])

    translator = Translator(tokenizer=tokenizer,transformer=model,start=dest_srt,end=dest_end)
    text,_,_ = translator(sentence = inputs,maxlen=64)

    #output_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return inputs,text


def text_to_speech(model,tokenizer,text):

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    
    output_array = output.detach().numpy()[0]
    sample_rate = model.config.sampling_rate

    return output_array,sample_rate


#print("model is loaded")


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

def get_mfcc(array,samplerate):

    #data,sample_rate = librosa.load(path,duration=30,offset=0.5)

    target_length = 30 * samplerate

    if array.ndim>1:
        array = array.flatten()

    if len(array) > target_length:
        array = array[:target_length]
    
    else:
        padding = target_length - len(array)
        
        array = np.pad(array,(0,padding),mode="constant")

    mfcc = np.mean(librosa.feature.mfcc(y=array ,sr = samplerate,n_mfcc=4000).T,axis=0)
    
    return mfcc


def get_language_voice(array,sample_rate):
    
    model = load_model(r"./my_latest_model.keras")

    category_list = {0:"hindi",1:"kannada",2:"tamil",3:"telugu"}
    lang_label = category_list[np.argmax(model.predict(np.expand_dims(np.expand_dims(np.array(get_mfcc(array=array,samplerate=sample_rate)),1),0)))]
    
    return lang_label


person1 = None
person2 = None
#both_assign = False

@app.route('/')
def index():
    return render_template('index.html')


"""@app.route('/assign_language/<person>')
def assign_language(person):
    global person1, person2, both_assign

    if detected_lang not in key_dict:
        return jsonify({"error": f"Language '{detected_lang}' not supported."}), 400

    if person == "person1":
        
        voice_array, sample_rate = record_voice()
        detected_lang = get_language_voice(array=voice_array, sample_rate=sample_rate)
        print(f"Detected language for {person}: {detected_lang}")
        person1 = key_dict[detected_lang]

    else:

        voice_array, sample_rate = record_voice()
        detected_lang = get_language_voice(array=voice_array, sample_rate=sample_rate)
        print(f"Detected language for {person}: {detected_lang}")
        person2 = key_dict[detected_lang]

    if person1 and person2:
        both_assign = True

    # Debugging the assignment
    print(f"Person 1: {person1}, Person 2: {person2}, Both assigned: {both_assign}")

    return jsonify({
        "person": person,
        "detected_language": detected_lang,
        "both_assigned": both_assign
    })

"""


@app.route('/select_language/<person>', methods=['POST'])
def select_language(person):
    global person1, person2

    data = request.get_json()
    selected_lang = data.get("lang")

    if selected_lang not in key_dict:
        return jsonify({"error": "Unsupported language selected."}), 400

    if person == "person1":
        if person2 is not None and selected_lang == person2[1]:
            return jsonify({"error": "Person 1 and Person 2 cannot have the same language. Please choose a different language."}), 400
        person1 = key_dict[selected_lang]
    elif person == "person2":
        if person1 is not None and selected_lang == person1[1]:
            return jsonify({"error": "Person 1 and Person 2 cannot have the same language. Please choose a different language."}), 400
        person2 = key_dict[selected_lang]

    return jsonify({
        "message": f"{person} language set to {selected_lang}",
        "both_assigned": person1 is not None and person2 is not None
    })


@app.route('/record/<person>',methods=['POST', 'GET'])
def record(person):

    global person1, person2

    #if not both_assigned:
        #return jsonify({"error": "Languages not yet assigned for both persons."}), 400


    if person1 is None or person2 is None:
        if person == "person1":
            voice_array, sample_rate = record_voice()
            detected_lang = get_language_voice(array=voice_array, sample_rate=sample_rate)
            print(f"Detected language for {person}: {detected_lang}")
            person1 = key_dict[detected_lang]
        else:
            voice_array, sample_rate = record_voice()
            detected_lang = get_language_voice(array=voice_array, sample_rate=sample_rate)
            print(f"Detected language for {person}: {detected_lang}")
            person2 = key_dict[detected_lang]
        return jsonify({
            "message": f"Language assigned for {person}. Waiting for the other person.",
            "detected_language": detected_lang
        })
    
    if person1 == person2:
        return jsonify({"error": "Both persons cannot have the same language. Please choose a different language."})

    if person1 and person2:
        if person == "person1":
            speech_scr = person1[0]
            src_str,src_end,tgt_str,tgt_end = person1[1], person1[2], person2[1], person2[2]
            tts_model, tts_tokenizer = person2[2],person2[3]
        else:
            speech_scr = person2[0]
            src_str,src_end,tgt_str,tgt_end = person2[1], person2[2], person1[1], person1[2]
            tts_model, tts_tokenizer =  person1[2],person1[3]
        

        voice_array, sample_rate = record_voice()
        #print(voice_array)
        text = speech_to_text(voice_array,sample_rate,src_lang=speech_scr)

        # Here is my Model How it will take ok

        input_text,output_text = text_translation(text=text,src_srt=src_str,src_end=src_end,dest_srt=tgt_str,dest_end=tgt_end)

        #translated = text_translation(text, src_lang=lang_src, dest_lang=lang_tgt)

        print(output_text)
        audio_data, audio_sr = text_to_speech(text = output_text, model=tts_model, tokenizer=tts_tokenizer)

        #text_to_speech(model=tam_text_speech_model,tokenizer=tam_text_speech_tokenizer,text=output_text)

        filename = f"{uuid.uuid4().hex}.wav"
        path = os.path.join(AUDIO_FOLDER, filename)
        torchaudio.save(path, torch.tensor(audio_data).unsqueeze(0), audio_sr)

        return jsonify({
        "text": text,
        "translated": output_text,
        "audio_url": f"/static/audio/{filename}"
        })

@app.route('/delete_audio/<filename>')
def delete_audio(filename):
    try:
        os.remove(os.path.join(AUDIO_FOLDER, filename))
        return "Deleted", 200
    except Exception as e:
        return str(e), 500


@app.route('/restart')
def restart_conversation():
    global person1, person2

    # Reset only the speaker state, not models
    person1 = None
    person2 = None
    #both_assign = False

    return render_template('index.html', message="Conversation restarted. Please reassign languages.")


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True) 