import streamlit as st
from PIL import Image
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "pranavpsv/genre-story-generator-v2"
TTS_MODEL = "Matthijs/mms-tts-eng"
DARK_WORDS = [
    "death",
    "dead",
    "darkness",
    "killed",
    "kidnapped",
    "incest",
    "deceased",
    "mysterious man",
]


st.set_page_config(page_title="Image to Audio Story", page_icon="📖")
st.title("📖 AI Storytelling App for Kids")
st.write("Upload an image or take a photo to create a short story with audio.")


def load_caption_pipeline():
    return pipeline("image-to-text", model=CAPTION_MODEL)


def load_story_pipeline():
    return pipeline("text-generation", model=STORY_MODEL)


def load_tts_pipeline():
    return pipeline("text-to-audio", model=TTS_MODEL)


def image_to_text(image):
    caption_pipe = load_caption_pipeline()
    result = caption_pipe(image)
    return result[0]["generated_text"]


def has_dark_content(story):
    story_lower = story.lower()
    for word in DARK_WORDS:
        if word in story_lower:
            return True
    return False


def finish_sentence(story):
    for mark in [".", "!", "?"]:
        position = story.rfind(mark)
        if position > 40:
            return story[: position + 1].strip()
    return story.strip() + "."


def fallback_story(text):
    return (
        f"One sunny day, {text} became the start of a happy adventure. "
        "The children laughed, shared their toys, and helped each other discover something new. "
        "A gentle breeze danced around them, and everyone felt brave and kind. "
        "When it was time to go home, they smiled and promised to play again tomorrow. 🌟"
    )


def text_to_story(text):
    story_pipe = load_story_pipeline()
    prompt = (
        "Write a warm and imaginative story for children aged 3 to 10. "
        "Use 50 to 100 words and add 1 to 3 suitable emoji naturally. "
        "The story must be happy, safe, and not scary. "
        "Do not mention death, darkness, violence, kidnapping, or adult topics. "
        f"Image description: {text}. Story:"
    )
    result = story_pipe(prompt, max_new_tokens=90, return_full_text=False)
    story = finish_sentence(result[0]["generated_text"].strip())

    if has_dark_content(story):
        story = fallback_story(text)

    return story


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
            scenario = image_to_text(image)
            story = text_to_story(scenario)
            st.session_state["scenario"] = scenario
            st.session_state["story"] = story

    if "scenario" in st.session_state:
        st.subheader("Image Caption")
        st.write(st.session_state["scenario"])

    if "story" in st.session_state:
        st.subheader("Generated Story ✨")
        st.write(st.session_state["story"])

    if "story" in st.session_state and st.button("Play Audio"):
        with st.spinner("Generating audio data..."):
            speech_output = story_to_audio(st.session_state["story"])
        audio_array = speech_output["audio"]
        sample_rate = speech_output["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
