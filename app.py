import io
import os
import tempfile
from flask import Flask, request, jsonify
import speech_recognition as sr
from pydub import AudioSegment

app = Flask(__name__)

# Define the route for the API endpoint

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No audio file provided.'}), 400

    # Check if actual_text parameter is provided
    if 'actual_text' in request.form:
        actual_text = request.form['actual_text']
        compute_accuracy = True
    else:
        actual_text = None
        compute_accuracy = False
    
    # Transcribe the audio using Sphinx
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = r.record(source)  # read the entire audio file
    text = r.recognize_sphinx(audio)

    # Compute accuracy if actual_text is provided
    if compute_accuracy:
        accuracy = compute_wer(text, actual_text)
        return jsonify({'text': text, 'accuracy': accuracy})
    else:
        return jsonify({'text': text})

# Compute word error rate (WER)
def compute_wer(hypothesis, reference):
    hypothesis = hypothesis.split()
    reference = reference.split()
    if len(reference) == 0:
        return 0 if len(hypothesis) == 0 else 1
    if len(hypothesis) == 0:
        return 1
    if len(reference) == 1 and len(hypothesis) == 1:
        return 0 if reference[0] == hypothesis[0] else 1

    # Compute Levenshtein distance
    distances = []
    for i in range(len(hypothesis) + 1):
        distances.append([i])
    for j in range(len(reference) + 1):
        distances[0].append(j)
    for i in range(1, len(hypothesis) + 1):
        for j in range(1, len(reference) + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                distances[i].append(distances[i - 1][j - 1])
            else:
                distances[i].append(
                    1 + min(distances[i - 1][j], distances[i][j - 1], distances[i - 1][j - 1]))

    # Compute WER
    wer = distances[-1][-1] / float(len(reference))
    return wer

if __name__ == '__main__':
    app.run(threaded=True, port=5000)