import streamlit as st
from PIL import Image
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"
TTS_MODEL = "Matthijs/mms-tts-eng"


st.set_page_config(page_title="Image to Audio Story", page_icon="📖")
st.title("📖 AI Storytelling App for Kids")
st.write("Upload an image or take a photo to create a short story with audio.")


def load_caption_pipeline():
    return pipeline("image-to-text", model=CAPTION_MODEL)


def load_story_pipeline():
    return pipeline("text2text-generation", model=STORY_MODEL)


def load_tts_pipeline():
    return pipeline("text-to-audio", model=TTS_MODEL)


def image_to_text(image):
    caption_pipe = load_caption_pipeline()
    result = caption_pipe(image)
    caption = result[0]["generated_text"]
    return clean_caption(caption)


def clean_caption(caption):
    caption = caption.replace(" illustration", "")
    caption = caption.replace(" drawing", "")
    caption = caption.replace(" cartoon", "")
    return caption.strip()


def expand_caption(caption):
    story_pipe = load_story_pipeline()
    prompt = (
        f"Describe this scene in one rich sentence for a children's story: {caption}. "
        "Mention the people, place, mood, and possible action."
    )
    result = story_pipe(prompt, max_new_tokens=60)
    description = result[0]["generated_text"].strip()

    if len(description.split()) < 8 or "describe this scene" in description.lower():
        description = f"{caption} in a cheerful place with a friendly and playful mood"

    return clean_caption(finish_sentence(description))


def finish_sentence(story):
    for mark in [".", "!", "?"]:
        position = story.rfind(mark)
        if position > 40:
            return story[: position + 1].strip()
    return story.strip() + "."


def simple_story(text):
    return (
        f"One sunny day, {text} became the start of a happy adventure. "
        "Everyone played together, shared kind words, and found something wonderful to smile about. "
        "A small surprise made the day feel special, and the children learned that imagination can turn any moment into magic. 🌟"
    )


def text_to_story(text):
    story_pipe = load_story_pipeline()
    prompt = (
        f"Tell a short happy children's story about this scene: {text}. "
        "Make it simple, kind, and imaginative."
    )
    result = story_pipe(
        prompt,
        max_new_tokens=120,
    )
    story = finish_sentence(result[0]["generated_text"].strip())

    if (
        "use 50 to 100 words" in story.lower()
        or "write a warm" in story.lower()
        or len(story.split()) < 30
    ):
        story = simple_story(text)

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
            caption = image_to_text(image)
            scenario = expand_caption(caption)
            story = text_to_story(scenario)
            speech_output = story_to_audio(story)
            st.session_state["scenario"] = scenario
            st.session_state["story"] = story
            st.session_state["audio_array"] = speech_output["audio"]
            st.session_state["sample_rate"] = speech_output["sampling_rate"]

    if "scenario" in st.session_state:
        st.subheader("Image Caption")
        st.write(st.session_state["scenario"])

    if "story" in st.session_state:
        st.subheader("Generated Story ✨")
        st.write(st.session_state["story"])

    if "audio_array" in st.session_state:
        st.subheader("Story Audio")
        st.audio(
            st.session_state["audio_array"],
            sample_rate=st.session_state["sample_rate"],
        )
