# AI Storytelling App for Kids

This repository contains a Streamlit storytelling app for ISOM5240 Assignment 1.

## Features

- Upload an image in PNG or JPG format
- Take a photo directly with a phone or webcam
- Generate an image caption with a Hugging Face model
- Turn the caption into a child-friendly 50-100 word story with a few emoji
- Convert the story to speech with a Hugging Face TTS model
- Play the audio in the browser

## Models and Tools

- Image captioning: `Salesforce/blip-image-captioning-base`
- Story generation: `pranavpsv/genre-story-generator-v2`
- Text-to-speech: `Matthijs/mms-tts-eng`
- UI framework: `Streamlit`

## Why This Version Is Better

- The code follows a simple classroom-style Streamlit structure.
- The app only uses the same core packages as the teacher's example: `streamlit`, `transformers`, `torch`, and `Pillow`.
- The app keeps the required assignment flow: image input, caption generation, story generation, and audio output.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Sign in to Streamlit Cloud.
3. Create a new app from this repository.
4. Set the main file path to `app.py`.
5. Deploy and use the public URL for submission.
