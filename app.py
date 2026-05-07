import streamlit as st
from PIL import Image
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "gpt2"
TTS_MODEL = "Matthijs/mms-tts-eng"


st.set_page_config(page_title="Image to Audio Story", page_icon="📖")
st.title("📖 AI Storytelling App for Kids")
st.write("Upload an image or take a photo to create a short story with audio.")


@st.cache_resource
def load_caption_pipeline():
    """Load the image captioning model."""
    return pipeline("image-to-text", model=CAPTION_MODEL)


@st.cache_resource
def load_story_pipeline():
    """Load the story generation model."""
    return pipeline("text-generation", model=STORY_MODEL)


@st.cache_resource
def load_tts_pipeline():
    """Load the text-to-speech model."""
    return pipeline("text-to-audio", model=TTS_MODEL)


def image_to_text(image):
    """Generate a caption from the uploaded image."""
    caption_pipe = load_caption_pipeline()
    result = caption_pipe(image)
    caption = result[0]["generated_text"]
    # Remove words that describe the image format rather than the scene
    for word in ["illustration", "drawing", "cartoon", "painting", "picture of"]:
        caption = caption.replace(word, "")
    return caption.strip()


def text_to_story(caption):
    """Generate a children's story (50-100 words) from the caption."""
    story_pipe = load_story_pipeline()
    prompt = f"Once upon a time, {caption}. "
    result = story_pipe(
        prompt,
        max_new_tokens=120,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
    )
    story = result[0]["generated_text"].strip()
    # Trim to the last complete sentence
    for mark in [".", "!", "?"]:
        pos = story.rfind(mark)
        if pos > 30:
            story = story[: pos + 1]
            break
    return story


def story_to_audio(story):
    """Convert the generated story text to audio."""
    tts_pipe = load_tts_pipeline()
    return tts_pipe(story)


# --- UI: Image source selection ---
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

    if st.button("Generate Story"):
        with st.spinner("Generating the story..."):
            caption = image_to_text(image)
            story = text_to_story(caption)
            speech_output = story_to_audio(story)
            st.session_state["caption"] = caption
            st.session_state["story"] = story
            st.session_state["audio_array"] = speech_output["audio"]
            st.session_state["sample_rate"] = speech_output["sampling_rate"]

    if "caption" in st.session_state:
        st.subheader("Image Caption")
        st.write(st.session_state["caption"])

    if "story" in st.session_state:
        st.subheader("Generated Story ✨")
        st.write(st.session_state["story"])

    if "audio_array" in st.session_state:
        st.subheader("Story Audio")
        st.audio(
            st.session_state["audio_array"],
            sample_rate=st.session_state["sample_rate"],
        )
