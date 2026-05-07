# AI Storytelling App for Kids

ISOM5240 Assignment — A Streamlit app that turns an image into an audio story for children aged 3–10.

## Features

- Upload an image or take a photo with your camera
- Generate an image caption using BLIP
- Generate a ~100-word children's story using GPT-2
- Convert the story to speech with MMS-TTS
- Play the audio directly in the browser

## Pipeline (3 Stages)

| Stage | Pipeline | Model |
|-------|----------|-------|
| Image to Text | `image-text-to-text` | `Salesforce/blip-image-captioning-base` |
| Text to Story | `text-generation` | `gpt2` |
| Story to Audio | `text-to-audio` | `Matthijs/mms-tts-eng` |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to `app.py`
4. Deploy
