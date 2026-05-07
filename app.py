# Program title: Storytelling App
# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline


# Function part
def img2text(image_path):
    image_to_text_model = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    image = Image.open(image_path)
    text = image_to_text_model(image)[0]["generated_text"]
    return text


# Main part
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

    # Stage 1: Image to Text (Using the function)
    st.text("Processing img2text... 🔍")
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (Inline, using GPT-2)
    st.text("Generating a story... ✨")
    story_pipe = pipeline("text-generation", model="gpt2")
    story_prompt = (
        f"Once upon a time 🌟, {scenario}. "
        "The children laughed in the sunshine ☀️, shared their toys 🧸, "
        "and followed a bright little butterfly 🦋 across the green grass 🌿. "
        "It led them to a hidden flower 🌸, so they made a wish together ⭐. "
        "Everyone went home smiling after a kind and happy adventure 🎉🌈❤️. "
    )
    story_results = story_pipe(story_prompt, max_new_tokens=50)[0]["generated_text"]
    st.write(f"**Story:** {story_results}")

    # Stage 3: Story to Audio (Inline)
    st.text("Generating audio data... 🔊")
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_results)

    # Play button
    if st.button("Play Audio ▶️"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
