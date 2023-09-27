from pydub import AudioSegment
import speech_recognition as sr
from joblib import Parallel, delayed

# Define function to handle recognition on each chunk
def recognize_chunk(chunk, recognizer, language):
    try:
        chunk.export("temp/temp.wav", format="wav")
        with sr.AudioFile("temp/temp.wav") as source:
            audio_text = recognizer.record(source)  
            text = recognizer.recognize_google(audio_text, language=language)
            return text
    except Exception as e:
        print(e)
        return ""

# Set the language
language = 'pt-BR'  # Portuguese (Portugal). Use 'pt-BR' for Portuguese (Brazil)

# Convert mp3 file to wav
sound = AudioSegment.from_mp3("docs/fipe.mp3")

# Break audio into 5-minute chunks
five_minutes = 5*60*1000  # In milliseconds
chunks = [sound[i:i + five_minutes] for i in range(0, len(sound), five_minutes)]
recognizer = sr.Recognizer()

# Process each chunk in parallel
outputs = Parallel(n_jobs=-1, verbose=10)(delayed(recognize_chunk)(chunk, recognizer, language) for chunk in chunks)

# Write output to file
with open('docs/video.txt', 'w') as f:
    for output in outputs:
        f.write(output)
