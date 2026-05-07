# Program title: Storytelling App
# Import part
import streamlit as st
from transformers import pipeline


# Function part
def img2text(url):
    image_to_text_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    text = image_to_text_model(url)[0]["generated_text"]
    return clean_text(text)


def clean_text(text):
    text = text.replace(" illustration", "")
    text = text.replace(" drawing", "")
    text = text.replace(" cartoon", "")
    return text.strip()


def finish_sentence(text):
    for mark in [".", "!", "?"]:
        position = text.rfind(mark)
        if position > 30:
            return text[: position + 1].strip()
    return text.strip() + "."


# Main part
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🤖")
st.header("ISOM5240: Turn Your Image to Audio Story")

# Image source selection: upload or camera
source = st.radio(
    "Choose an image source",
    ["Upload an image", "Take a photo"],
    horizontal=True,
)

uploaded_file = None
if source == "Upload an image":
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture with your camera")

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text (Using the function)
    st.text("Processing img2text...")
    scenario = img2text(uploaded_file.name)
    rich_scenario = f"{scenario} in a cheerful place with a friendly and playful mood"
    st.write(f"**Scenario:** {finish_sentence(rich_scenario)}")

    # Stage 2: Text to Story (Inline)
    st.text("Generating a story...")
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_prompt = (
        "Write only a short, happy story for children. "
        "No movie names. No actor names. No scary events. "
        "The story should be friendly, simple, and imaginative. "
        f"Scene: {rich_scenario}. Story:"
    )
    story_results = story_pipe(
        story_prompt,
        max_new_tokens=100,
        return_full_text=False,
    )
    story = finish_sentence(story_results[0]["generated_text"])
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio (Inline)
    st.text("Generating audio data...")
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
