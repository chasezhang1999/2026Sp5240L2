import hashlib
from io import BytesIO
from typing import Optional

import streamlit as st
from gtts import gTTS
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"
MIN_WORDS = 50
MAX_WORDS = 100


st.set_page_config(
    page_title="AI Storytelling App for Kids",
    page_icon="📖",
    layout="centered",
)


def load_caption_pipeline():
    """Load the captioning model."""
    return pipeline("image-to-text", model=CAPTION_MODEL)


def load_story_pipeline():
    """Load the story generation model."""
    return pipeline("text2text-generation", model=STORY_MODEL)


def generate_caption(image: Image.Image) -> str:
    """Generate a short caption from the uploaded image."""
    caption_pipeline = load_caption_pipeline()
    result = caption_pipeline(image)
    return result[0]["generated_text"].strip().capitalize()


def remove_repeated_sentences(story: str) -> str:
    """Remove duplicate sentences from model output."""
    raw_sentences = [
        sentence.strip()
        for sentence in story.replace("!", ".").replace("?", ".").split(".")
        if sentence.strip()
    ]

    unique_sentences = []
    seen = set()

    for sentence in raw_sentences:
        normalized = " ".join(sentence.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_sentences.append(sentence)

    if not unique_sentences:
        return story.strip()

    return ". ".join(unique_sentences).strip() + "."


def trim_story_to_limit(story: str, max_words: int = MAX_WORDS) -> str:
    """Keep the story inside the assignment word limit."""
    words = story.split()
    if len(words) <= max_words:
        return story.strip()
    return " ".join(words[:max_words]).strip(" ,.;:") + "."


def expand_short_story(story: str, caption: str, min_words: int = MIN_WORDS) -> str:
    """Pad a short story so it still meets the assignment requirement."""
    if len(story.split()) >= min_words:
        return story.strip()

    extra_sentences = [
        f"In this little world, {caption.lower()} made everyone feel curious and happy.",
        "Soon, a gentle surprise turned the moment into a fun adventure full of smiles.",
        "When the day ended, everyone learned that kindness and imagination make every story brighter.",
    ]

    updated_story = story.strip()
    for sentence in extra_sentences:
        if len(updated_story.split()) >= min_words:
            break
        updated_story = f"{updated_story} {sentence}".strip()

    return updated_story


def generate_story(caption: str) -> str:
    """Generate a child-friendly story from the image caption."""
    story_pipeline = load_story_pipeline()
    prompt = (
        "Write a warm and imaginative story for children aged 3 to 10. "
        f"Use only {MIN_WORDS} to {MAX_WORDS} words. "
        "Write 4 to 6 short sentences in simple English. "
        "Add 1 to 3 suitable emoji naturally in the story. "
        "Do not repeat the same idea or sentence. "
        f"Base the story on this image description: {caption}"
    )

    result = story_pipeline(
        prompt,
        max_new_tokens=140,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        no_repeat_ngram_size=3,
    )
    story = result[0]["generated_text"].strip()
    story = remove_repeated_sentences(story)
    story = expand_short_story(story, caption)
    story = trim_story_to_limit(story)
    return story


def text_to_speech(text: str, language: str = "en") -> bytes:
    """Convert text to MP3 bytes for playback and download."""
    audio_buffer = BytesIO()
    tts = gTTS(text=text, lang=language, slow=False)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()


def build_story(image: Image.Image) -> tuple[str, str, bytes]:
    """Run the full image -> caption -> story -> audio pipeline."""
    caption = generate_caption(image)
    story = generate_story(caption)
    audio_bytes = text_to_speech(story)
    return caption, story, audio_bytes


def get_upload_signature(file_bytes: bytes) -> str:
    """Create a stable signature for the uploaded image."""
    return hashlib.sha1(file_bytes).hexdigest()


def reset_outputs_on_new_upload(upload_signature: Optional[str]) -> None:
    """Clear previous outputs when the uploaded image changes."""
    previous_signature = st.session_state.get("last_upload_signature")
    if upload_signature and upload_signature != previous_signature:
        st.session_state.pop("caption", None)
        st.session_state.pop("story", None)
        st.session_state.pop("audio_bytes", None)
        st.session_state["last_upload_signature"] = upload_signature


def load_uploaded_image(uploaded_file) -> tuple[Image.Image, bytes]:
    """Read uploaded bytes and build a display-ready image."""
    file_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image, file_bytes


def main():
    st.title("AI Storytelling App for Kids 📖")
    st.write(
        "Upload a picture or take a photo, and the app will create a short story, then read it aloud."
    )
    st.caption(
        "The app uses Hugging Face for image captioning and story generation, and gTTS for audio."
    )

    source = st.radio(
        "Choose an image source",
        ["Upload an image", "Take a photo"],
        horizontal=True,
    )

    uploaded_file = None
    if source == "Upload an image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
        )
    else:
        uploaded_file = st.camera_input("Take a picture with your camera")

    if uploaded_file is None:
        st.info("Please upload a PNG or JPG image, or take a photo to begin.")
        return

    try:
        image, file_bytes = load_uploaded_image(uploaded_file)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please try a PNG or JPG file.")
        return

    reset_outputs_on_new_upload(get_upload_signature(file_bytes))

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Generate Story", type="primary"):
        try:
            with st.spinner("Creating a story and audio for you..."):
                caption, story, audio_bytes = build_story(image)
                st.session_state["caption"] = caption
                st.session_state["story"] = story
                st.session_state["audio_bytes"] = audio_bytes
        except Exception as error:
            st.error(f"An error occurred while generating the story: {error}")

    if "caption" in st.session_state:
        st.subheader("Image Caption")
        st.write(st.session_state["caption"])

    if "story" in st.session_state:
        st.subheader("Generated Story ✨")
        st.write(st.session_state["story"])
        st.download_button(
            "Download Story as Text",
            data=st.session_state["story"],
            file_name="story.txt",
            mime="text/plain",
        )

    if "audio_bytes" in st.session_state:
        st.subheader("Story Audio")
        st.audio(st.session_state["audio_bytes"], format="audio/mp3")
        st.download_button(
            "Download Story Audio",
            data=st.session_state["audio_bytes"],
            file_name="story.mp3",
            mime="audio/mpeg",
        )


if __name__ == "__main__":
    main()
