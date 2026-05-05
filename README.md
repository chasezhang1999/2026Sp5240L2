# AI Storytelling App for Kids

This repository contains a Streamlit storytelling app for ISOM5240 Assignment 1.

## Features

- Upload an image in PNG or JPG format
- Generate an image caption with a Hugging Face model
- Turn the caption into a child-friendly 50-100 word story
- Convert the story to speech
- Play the audio in the browser
- Download the generated story as text or MP3

## Models and Tools

- Image captioning: `Salesforce/blip-image-captioning-base`
- Story generation: `google/flan-t5-small`
- Text-to-speech: `gTTS`
- UI framework: `Streamlit`

## Why This Version Is Better

- Hugging Face pipelines are cached, so the app does not reload models every time.
- The app processes uploaded images in memory instead of writing temporary files to disk.
- Story output is cleaned to reduce repeated sentences.
- The generated story is kept within the assignment word limit.
- Audio is cached and can be replayed or downloaded.

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
