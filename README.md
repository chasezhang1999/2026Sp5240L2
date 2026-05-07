# AI Storytelling App for Kids

This project is a Streamlit storytelling application for ISOM5240 Assignment 1.

## Features

- Upload an image
- Generate an image caption with a Hugging Face image model
- Generate a 50-100 word story for children
- Convert the story into speech
- Reuse loaded models with Streamlit caching
- Download the generated story as text or MP3 audio

## Models and Tools

- Image captioning model: `Salesforce/blip-image-captioning-base`
- Story generation model: `google/flan-t5-small`
- Text-to-speech: `gTTS`
- Web framework: `Streamlit`

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:

```bash
streamlit run app.py
```

## Optimization Notes

- The app caches Hugging Face pipelines so the models are loaded only once per session.
- Uploaded images are tracked by file content instead of filename, which prevents stale results when a different image has the same name.
- The story generation step removes repeated sentences and keeps the final output within the assignment word limit.
- Generated audio is cached for repeated playback and download.

## Streamlit Cloud Deployment

1. Upload this folder to a GitHub repository.
2. Sign in to Streamlit Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy and copy the public URL for submission.

## Submission Checklist

- `app.py`
- `requirements.txt`
- Other required files
- Streamlit Cloud URL
