import time
import logging
import numpy as np
import queue
import requests
import io
import soundfile as sf
import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import csv
from datetime import datetime
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import sys
import codecs
from typing import List, Dict
import torch
import sounddevice as sd
from silero_vad import get_speech_timestamps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

logging.info("Imports completed in %.2f seconds", time.time() - start_time)

logging.info("Setting up stdout encoding...")
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
logging.info("Stdout encoding set up")

logging.info("Loading environment variables...")
load_dotenv()
logging.info("Environment variables loaded")

logging.info("Initializing OpenAI client...")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.info("OpenAI client initialized")

logging.info("Initializing ElevenLabs client...")
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
logging.info("ElevenLabs client initialized")

logging.info("Setting up constants...")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_ENDPOINT = "https://mrholocg8pxhkacd.us-east-1.aws.endpoints.huggingface.cloud"
ELEVENLABS_VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"
SAMPLE_RATE = 16000
CHANNELS = 1
logging.info("Constants set up")

logging.info("Initializing audio queue...")
audio_queue = queue.Queue()
logging.info("Audio queue initialized")

logging.info("Loading Silero VAD model...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
logging.info("Silero VAD model loaded")

conversation_history: List[Dict[str, str]] = []

def measure_latency(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Starting {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logging.info(f"Finished {func.__name__} in {latency:.2f} ms")
        return result, latency
    return wrapper

def display_arabic(text):
    logging.info("Displaying Arabic text")
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        logging.info("Arabic text displayed successfully")
        return bidi_text
    except Exception as e:
        logging.error(f"Error reshaping text: {e}")
        return text

def audio_callback(indata, frames, time, status):
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())

def record_audio_with_vad():
    logging.info("Starting audio recording with VAD")
    print("Listening... Speak when ready.")
    audio_queue.queue.clear()
    
    vad_start_time = time.time()
    total_processing_time = 0
    num_chunks_processed = 0
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        audio_data = []
        silence_duration = 0
        speech_detected = False
        
        while True:
            if not audio_queue.empty():
                chunk_start_time = time.time()
                chunk = audio_queue.get()
                audio_data.extend(chunk.flatten())
                
                if len(audio_data) > SAMPLE_RATE:  # Process in 1-second chunks
                    audio_tensor = torch.tensor(audio_data[-SAMPLE_RATE:])
                    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=SAMPLE_RATE)
                    
                    if speech_timestamps and not speech_detected:
                        logging.info("Speech detected, recording started")
                        print("Recording... Please speak.")
                        speech_detected = True
                        silence_duration = 0
                    elif not speech_timestamps and speech_detected:
                        silence_duration += 1
                        if silence_duration > 2:  # Stop after 2 seconds of silence
                            logging.info("Recording stopped")
                            break
                    else:
                        silence_duration = 0
                
                chunk_end_time = time.time()
                chunk_processing_time = chunk_end_time - chunk_start_time
                total_processing_time += chunk_processing_time
                num_chunks_processed += 1
            
            sd.sleep(10)  # Small sleep to prevent busy-waiting
    
    vad_end_time = time.time()
    total_vad_time = vad_end_time - vad_start_time
    avg_chunk_processing_time = total_processing_time / num_chunks_processed if num_chunks_processed > 0 else 0
    
    logging.info(f"VAD total time: {total_vad_time:.2f} seconds")
    logging.info(f"Average chunk processing time: {avg_chunk_processing_time*1000:.2f} ms")
    
    return np.array(audio_data, dtype=np.float32), total_vad_time, avg_chunk_processing_time

@measure_latency
def transcribe_audio(audio):
    logging.info("Transcribing audio")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "audio/flac"
    }
    
    audio = np.frombuffer(audio, dtype=np.float32) if not isinstance(audio, np.ndarray) else audio
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format='flac')
    flac_data = buffer.getvalue()
    
    try:
        logging.info("Sending request to Hugging Face API")
        response = requests.post(HUGGINGFACE_ENDPOINT, headers=headers, data=flac_data)
        response.raise_for_status()
        logging.info("Received response from Hugging Face API")
        return response.json().get("text")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in transcription: {e}")
        return None

@measure_latency
def get_llm_response(transcription: str) -> str:
    global conversation_history
    logging.info("Getting LLM response")
    if transcription is None:
        logging.warning("Transcription is None, returning error message")
        return "I'm sorry, I couldn't transcribe the audio. Could you please try again?"
    
    conversation_history.append({"role": "user", "content": transcription})
    
    system_message = """إنت مساعد صوتي لشركة Sphinx تورز، وهي وكالة سفر موجودة في القاهرة، مصر. الشركة متخصصة في تنظيم الرحلات والإجازات، وبتقدم تجارب سفر مخصصة للعملاء. Sphinx تورز شغالة من الساعة 9 الصبح لحد الساعة 6 بليل من الاثنين للسبت، ومقفولين يوم الأحد. وظيفتك الأساسية هي الرد على الأسئلة عن باقات السفر وحجز الرحلات. لما الcaller يكون عاوز يحجز رحلة، هدفك هو جمع كل المعلومات اللازمة بشكل فعال مع الحفاظ على نبرة ودية وجذابة:
1. اسأل عن اسمه بالكامل.
2. اسأل هو عايز يسافر فين أو أي باقة رحلات مهتم بيها.
3. اطلب منه مواعيد السفر المفضلة عنده.
4. أكد على كل التفاصيل مع الcaller، بما في ذلك الوجهة والتواريخ وأي طلبات خاصة.
5. خلي نبرتك خفيفة ومرحة، وضيف شوية هزار.
6. استخدم لغة عادية وحوارية. ما تترددش تقول حاجات زي "اممم..."، "طيب..."، أو "يعني" عشان تخلي الكلام طبيعي.
7. ده حوار صوتي، فخليك مختصر وفي الصميم. ما تطولش في الشرح."""

    try:
        logging.info("Sending request to OpenAI API")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                *conversation_history
            ],
            temperature=0.3,
            max_tokens=250
        )
        logging.info("Received response from OpenAI API")
        assistant_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response
    except Exception as e:
        logging.error(f"Error in LLM response: {e}")
        return "I'm sorry, I encountered an error. Could you please try again?"

@measure_latency
def text_to_speech(text: str) -> IO[bytes]:
    logging.info("Converting text to speech")
    try:
        logging.info("Sending request to ElevenLabs API")
        response = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            output_format="mp3_22050_32",
            text=text,
            #language_code='ar',
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        logging.info("Received response from ElevenLabs API")

        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)

        logging.info("Playing audio")
        audio_segment = AudioSegment.from_mp3(audio_stream)
        audio_array = np.array(audio_segment.get_array_of_samples())
        sd.play(audio_array, samplerate=audio_segment.frame_rate)
        sd.wait()
        logging.info("Audio playback completed")

        return audio_stream
    except Exception as e:
        logging.error(f"Error in text-to-speech conversion: {e}")
        return None

def write_to_csv(data, filename="voice_assistant_log.csv"):
    logging.info(f"Writing data to CSV file: {filename}")
    fieldnames = [
        "timestamp", "turn", "transcription", "llm_response", 
        "transcription_latency", "llm_latency", "tts_latency", 
        "total_latency", "highest_latency"
    ]
    file_exists = os.path.isfile(filename)
    
    try:
        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            # Sanitize the data
            sanitized_data = {
                k: v.replace('\n', ' ').replace('\r', '') if isinstance(v, str) else v
                for k, v in data.items()
            }
            
            writer.writerow(sanitized_data)
        logging.info("Data written to CSV successfully")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")

def is_goodbye(text: str) -> bool:
    goodbye_phrases = [
        "مع السلامة", "وداعا", "باي", "سلام", "تصبح على خير", "إلى اللقاء",
        "شكرا", "شكرًا", "خلاص كده"
    ]
    return any(phrase in text.lower() for phrase in goodbye_phrases)

@measure_latency
def get_initial_greeting() -> str:
    global conversation_history
    logging.info("Getting initial greeting from LLM")
    
    system_message = """إنت مساعد صوتي لشركة Sphinx تورز، وهي وكالة سفر موجودة في القاهرة، مصر. الشركة متخصصة في تنظيم الرحلات والإجازات، وبتقدم تجارب سفر مخصصة للعملاء. Sphinx تورز شغالة من الساعة 9 الصبح لحد الساعة 6 بليل من الاثنين للسبت، ومقفولين يوم الأحد. وظيفتك الأساسية هي الرد على الأسئلة عن باقات السفر وحجز الرحلات. قدم ترحيب قصير وودي باللهجة المصرية."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "ابدأ المحادثة بترحيب قصير وودي."}
            ],
            temperature=0.7,
            max_tokens=50
        )
        greeting = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": greeting})
        return greeting
    except Exception as e:
        logging.error(f"Error getting initial greeting: {e}")
        return "أهلا بيك في Sphinx تورز! إزاي أقدر أساعدك النهاردة؟"  # Fallback

def main():
    global conversation_history
    logging.info("Starting main function")
    turn = 0
    conversation_history = []  # Initialize new conversation

    # Get and play initial greeting
    initial_greeting, greeting_latency = get_initial_greeting()
    logging.info(f"Initial Greeting: {display_arabic(initial_greeting)}")
    tts_result, tts_latency = text_to_speech(initial_greeting)

    while True:
        turn += 1
        logging.info(f"Starting turn {turn}")
        print("\nListening for speech...")
        audio_data, vad_total_time, vad_avg_chunk_time = record_audio_with_vad()
        logging.info("Audio recorded")

        overall_start_time = time.time()

        transcription, transcription_latency = transcribe_audio(audio_data)
        logging.info(f"Transcription: {display_arabic(transcription)}")

        if is_goodbye(transcription):
            goodbye_message = "شكرًا لاستخدامك Sphinx تورز. نتمنى لك يومًا سعيدًا!"
            logging.info(f"Goodbye message: {display_arabic(goodbye_message)}")
            tts_result, tts_latency = text_to_speech(goodbye_message)
            print("Ending conversation. Goodbye!")
            break

        llm_response, llm_latency = get_llm_response(transcription)
        logging.info(f"LLM Response: {display_arabic(llm_response).encode('utf-8').decode('utf-8')}")
        
        logging.info("Converting response to speech")
        tts_result, tts_latency = text_to_speech(llm_response)

        overall_latency = (time.time() - overall_start_time) * 1000

        latencies = {
            "VAD Total": vad_total_time * 1000,  # Convert to ms
            "VAD Avg Chunk": vad_avg_chunk_time * 1000,  # Convert to ms
            "Transcription": transcription_latency,
            "LLM": llm_latency,
            "Text-to-Speech": tts_latency
        }
        highest_latency = max(latencies, key=latencies.get)

        data = {
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "transcription": transcription,
            "llm_response": llm_response,
            "vad_total_latency": vad_total_time * 1000,
            "vad_avg_chunk_latency": vad_avg_chunk_time * 1000,
            "transcription_latency": transcription_latency,
            "llm_latency": llm_latency,
            "tts_latency": tts_latency,
            "total_latency": overall_latency,
            "highest_latency": highest_latency
        }

        write_to_csv(data)

        logging.info(f"Overall latency: {overall_latency:.2f} ms")
        logging.info(f"Highest latency: {highest_latency} ({latencies[highest_latency]:.2f} ms)")

    logging.info("Main function completed")

if __name__ == "__main__":
    logging.info("Script started")
    main()
    logging.info("Script completed")