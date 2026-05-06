import streamlit as st
from PIL import Image
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-base"
TTS_MODEL = "Matthijs/mms-tts-eng"


st.set_page_config(page_title="Image to Audio Story", page_icon="📖")
st.title("📖 AI Storytelling App for Kids")
st.write("Upload an image or take a photo to create a short story with audio.")


@st.cache_resource
def load_caption_pipeline():
    return pipeline("image-to-text", model=CAPTION_MODEL)


@st.cache_resource
def load_story_pipeline():
    return pipeline("text2text-generation", model=STORY_MODEL)


@st.cache_resource
def load_tts_pipeline():
    return pipeline("text-to-audio", model=TTS_MODEL)


def image_to_text(image):
    caption_pipe = load_caption_pipeline()
    result = caption_pipe(image)
    caption = result[0]["generated_text"]
    # Remove words that describe the image format rather than the scene
    for word in ["illustration", "drawing", "cartoon", "painting", "picture of"]:
        caption = caption.replace(word, "")
    return caption.strip()


def text_to_story(caption):
    story_pipe = load_story_pipeline()
    prompt = (
        f"Tell a short happy children's story about this scene: {caption}. "
        "Make it simple, kind, and imaginative."
    )
    result = story_pipe(prompt, max_new_tokens=120)
    return result[0]["generated_text"].strip()


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
