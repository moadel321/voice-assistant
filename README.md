# Voice Assistant Project

This project implements an advanced voice assistant that can transcribe speech to text using a Hugging Face inference endpoint, generate responses using OpenAI's GPT-4, and convert text back to speech using OpenAI's TTS API.

## Features
- Audio recording using the spacebar as a push-to-talk mechanism
- Speech-to-text transcription using a Hugging Face inference endpoint
- Text generation using OpenAI's GPT-4 model
- Text-to-speech output using OpenAI's TTS API
- Proper display and handling of Arabic text (right-to-left)
- Comprehensive latency logging for performance analysis
- CSV logging of conversation turns and latencies

## Requirements
- Python 3.7.1 or newer
- Required Python packages: openai, sounddevice, numpy, requests, soundfile, arabic-reshaper, python-bidi, python-dotenv, pydub, keyboard

## Setup
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Set up your environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token
   - `ELEVENLABS_API_KEY`: Your ElevenLabs API key (if using ElevenLabs for TTS)
4. Ensure you have FFmpeg installed and available in your system PATH

## Usage
1. Run the script: `python voice_assist_poc.py`
2. Press and hold the spacebar to speak
3. Release the spacebar to process the audio and get a response
4. The assistant will transcribe your speech, generate a response, and speak it back to you
5. Press 'q' to quit the program

## Performance Logging
The script logs detailed performance metrics for each turn of the conversation, including:
- Transcription latency
- LLM response latency
- Text-to-speech latency
- Overall latency
- Highest latency component

These metrics are saved to a CSV file for further analysis.

## Customization
- You can adjust the `SAMPLE_RATE` and `CHANNELS` constants to modify audio recording settings
- The Hugging Face endpoint URL can be changed by modifying the `HUGGINGFACE_ENDPOINT` constant
- GPT-4 parameters such as temperature can be adjusted in the `get_llm_response` function

## Troubleshooting
- If you encounter issues with Arabic text display, ensure your console supports UTF-8 encoding
- For audio playback issues, check that your system's audio output is correctly configured

## Future Improvements
- Check issues 

## Contributing
Contributions to improve the voice assistant are welcome. Please fork the repository and submit a pull request with your changes.

