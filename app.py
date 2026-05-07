# Program title: Storytelling App
# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline

# ------------------ Parameters ------------------
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
AUDIO_MODEL = "Matthijs/mms-tts-eng"


# ------------------ Functions ------------------
def img2text(image_path):
    image_to_text_model = pipeline("image-text-to-text", model=CAPTION_MODEL)
    image = Image.open(image_path)
    text = image_to_text_model(image, text="a picture of")[0]["generated_text"]
    return text


def finish_sentence(text):
    """Trim generated text to the last complete sentence."""
    for mark in [".", "!", "?"]:
        pos = text.rfind(mark)
        if pos > 30:
            return text[: pos + 1].strip()
    return text.strip() + "."


# ------------------ Main ------------------
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🤖")
st.header("ISOM5240: Turn Your Image to Audio Story")

# Image source: upload or camera
source = st.radio("Choose an image source", ["Upload an image 📁", "Take a photo 📷"], horizontal=True)

uploaded_file = None
if source == "Upload an image 📁":
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture with your camera")

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image 🖼️", use_column_width=True)

    # Stage 1: Image to Text
    st.text("Processing img2text...")
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (~100 words for kids aged 3-10)
    st.text("Generating a story...")
    story_pipe = pipeline("text-generation", model=STORY_MODEL)
    story_prompt = (
        f"Write a warm and happy children's story for kids aged 3 to 10. "
        f"The story should be about 100 words and based on: {scenario}. "
        f"Use simple English and short sentences. "
    )
    story_raw = story_pipe(
        story_prompt,
        max_new_tokens=140,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        no_repeat_ngram_size=3,
    )[0]["generated_text"]
    story = finish_sentence(story_raw)
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio
    st.text("Generating audio data...")
    audio_pipe = pipeline("text-to-audio", model=AUDIO_MODEL)
    audio_data = audio_pipe(story)

    # Play button
    if st.button("Play Audio ▶️"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
