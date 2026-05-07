# Program title: Storytelling App
# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "gpt2"
AUDIO_MODEL = "Matthijs/mms-tts-eng"


# Function part
def img2text(image_path):
    image_to_text_model = pipeline("image-text-to-text", model=CAPTION_MODEL)
    image = Image.open(image_path)
    text = image_to_text_model(image, text="a picture of")[0]["generated_text"]
    return text


def finish_sentence(text):
    for mark in [".", "!", "?"]:
        pos = text.rfind(mark)
        if pos > 30:
            return text[: pos + 1].strip()
    return text.strip() + "."


# Main part
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🤖")
st.header("ISOM5240: Turn Your Image to Audio Story")

source = st.radio("Choose an image source", ["Upload an image 📁", "Take a photo 📷"], horizontal=True)

uploaded_file = None
if source == "Upload an image 📁":
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture with your camera")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image 🖼️", use_column_width=True)

    # Detect new upload to avoid re-running on button click
    file_key = hash(bytes_data)
    if st.session_state.get("file_key") != file_key:
        st.session_state["file_key"] = file_key

        # Stage 1: Image to Text
        with st.spinner("Processing img2text..."):
            st.session_state["scenario"] = img2text(uploaded_file.name)

        # Stage 2: Text to Story
        with st.spinner("Generating a story..."):
            story_pipe = pipeline("text-generation", model=STORY_MODEL)
            story_prompt = (
                f"Once upon a time, {st.session_state['scenario']}. "
                "The sun was shining and a gentle breeze blew through the trees. "
                "The children ran across the green meadow, chasing butterflies and picking flowers. "
                "Near the old oak tree, they found a tiny door hidden in the roots. "
            )
            story_raw = story_pipe(
                story_prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.75,
                top_p=0.85,
                no_repeat_ngram_size=3,
                return_full_text=False,
            )[0]["generated_text"]
            st.session_state["story"] = finish_sentence(story_raw)

        # Stage 3: Story to Audio
        with st.spinner("Generating audio data..."):
            audio_pipe = pipeline("text-to-audio", model=AUDIO_MODEL)
            st.session_state["audio_data"] = audio_pipe(st.session_state["story"])

    # Display cached results
    st.write(f"**Scenario:** {st.session_state['scenario']}")
    st.write(f"**Story:** {st.session_state['story']}")

    if st.button("Play Audio ▶️"):
        audio_array = st.session_state["audio_data"]["audio"]
        sample_rate = st.session_state["audio_data"]["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
