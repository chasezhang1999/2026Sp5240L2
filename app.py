import streamlit as st
from PIL import Image
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "pranavpsv/genre-story-generator-v2"
TTS_MODEL = "Matthijs/mms-tts-eng"


st.set_page_config(page_title="Image to Audio Story", page_icon="📖")
st.title("📖 AI Storytelling App for Kids")
st.write("Upload an image or take a photo to create a short story with audio.")


@st.cache_resource
def load_caption_pipeline():
    return pipeline("image-to-text", model=CAPTION_MODEL)


@st.cache_resource
def load_story_pipeline():
    return pipeline("text-generation", model=STORY_MODEL)


@st.cache_resource
def load_tts_pipeline():
    return pipeline("text-to-audio", model=TTS_MODEL)


def image_to_text(image):
    caption_pipe = load_caption_pipeline()
    result = caption_pipe(image)
    return result[0]["generated_text"]


def text_to_story(text):
    story_pipe = load_story_pipeline()
    prompt = (
        "Write a warm and imaginative story for children aged 3 to 10. "
        "Use 50 to 100 words and add 1 to 3 suitable emoji naturally. "
        f"Image description: {text}. Story:"
    )
    result = story_pipe(prompt, max_new_tokens=120)
    return result[0]["generated_text"]


def story_to_audio(story):
    tts_pipe = load_tts_pipeline()
    return tts_pipe(story)


source = st.radio(
    "Choose an image source",
    ["Upload an image", "Take a photo"],
    horizontal=True,
)

uploaded_file = None
if source == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture with your camera")


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating the story..."):
        scenario = image_to_text(image)
        story = text_to_story(scenario)
        speech_output = story_to_audio(story)

    st.subheader("Image Caption")
    st.write(scenario)

    st.subheader("Generated Story ✨")
    st.write(story)

    if st.button("Play Audio"):
        audio_array = speech_output["audio"]
        sample_rate = speech_output["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
