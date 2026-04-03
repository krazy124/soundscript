import io
import os
import re
import html
import tempfile
from urllib.parse import urlparse, parse_qs

import streamlit as st
import whisper

from moviepy.video.io.VideoFileClip import VideoFileClip
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Video to Text Transcriber",
    layout="wide"
)


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_whisper_model(model_name: str):
    """
    Load Whisper once and cache it so it doesn't reload on every rerun.
    """
    return whisper.load_model(model_name)


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """
    Extract audio from a local video file and save it as MP3.
    """
    clip = None
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise ValueError("This video does not appear to contain an audio track.")

        clip.audio.write_audiofile(
            output_audio_path,
            codec="mp3",
            logger=None
        )
    finally:
        if clip is not None:
            clip.close()


def make_pdf_bytes(title: str, body_text: str) -> bytes:
    """
    Create a simple PDF in memory and return its bytes.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    left_margin = 50
    top_margin = height - 50
    line_height = 16
    max_width_chars = 95

    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, top_margin, title)

    pdf.setFont("Helvetica", 10)
    y = top_margin - 30

    paragraphs = body_text.split("\n")

    for paragraph in paragraphs:
        if not paragraph.strip():
            y -= line_height
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = top_margin
            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= max_width_chars:
                current_line = test_line
            else:
                pdf.drawString(left_margin, y, current_line)
                y -= line_height

                if y < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = top_margin

                current_line = word

        if current_line:
            pdf.drawString(left_margin, y, current_line)
            y -= line_height

            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = top_margin

    pdf.save()
    buffer.seek(0)
    return buffer.read()


def safe_filename(name: str) -> str:
    """
    Make a file-safe base name.
    """
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "transcript"


def copy_to_clipboard_html(text: str):
    """
    Render a browser-side copy button using JavaScript.
    """
    escaped_text = html.escape(text).replace("\n", "\\n").replace("'", "\\'")
    button_html = f"""
    <div style="margin-top: 0.5rem; margin-bottom: 1rem;">
        <button
            onclick="navigator.clipboard.writeText('{escaped_text}')"
            style="
                background-color:#4CAF50;
                color:white;
                border:none;
                padding:10px 16px;
                border-radius:8px;
                cursor:pointer;
                font-size:14px;
            "
        >
            Copy transcript to clipboard
        </button>
    </div>
    """
    st.components.v1.html(button_html, height=55)


def is_valid_youtube_url(url: str) -> bool:
    """
    Basic YouTube URL validation.
    """
    pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"
    return bool(re.match(pattern, url.strip()))


def extract_youtube_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from common YouTube URL formats.
    """
    parsed = urlparse(url)

    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/")
        if video_id:
            return video_id

    if parsed.netloc in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]

        if parsed.path.startswith("/shorts/"):
            parts = parsed.path.split("/")
            if len(parts) >= 3 and parts[2]:
                return parts[2]

        if parsed.path.startswith("/embed/"):
            parts = parsed.path.split("/")
            if len(parts) >= 3 and parts[2]:
                return parts[2]

    raise ValueError("Could not extract a YouTube video ID from that URL.")

def fetch_youtube_captions(youtube_url: str):
    """
    Fetch YouTube captions (compatible with Streamlit Cloud version).
    """
    video_id = extract_youtube_video_id(youtube_url)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        raise ValueError("No captions available for this video.")

    transcript_text = " ".join([item["text"] for item in transcript]).strip()

    if not transcript_text:
        raise ValueError("Captions were found, but no usable text was returned.")

    fallback_title = f"youtube_{video_id}"
    return transcript_text, fallback_title

def reset_transcript_state():
    st.session_state.transcript = ""
    st.session_state.transcription_complete = False
    st.session_state.source_label = ""
    st.session_state.file_base = "transcript"
    st.session_state.original_name = "Transcript"


# =========================
# SESSION STATE
# =========================
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "transcription_complete" not in st.session_state:
    st.session_state.transcription_complete = False

if "source_label" not in st.session_state:
    st.session_state.source_label = ""

if "file_base" not in st.session_state:
    st.session_state.file_base = "transcript"

if "original_name" not in st.session_state:
    st.session_state.original_name = "Transcript"


# =========================
# UI
# =========================
st.title("Video to Text Transcriber")
st.write("Upload a video file for Whisper transcription, or paste a YouTube link to pull captions.")

with st.expander("Settings", expanded=True):
    model_name = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Used for uploaded video files. Smaller models are faster. Larger models may be more accurate."
    )

input_mode = st.radio(
    "Choose input source",
    options=["Upload Video File", "YouTube Link"],
    horizontal=True
)

if input_mode == "Upload Video File":
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv", "mpeg", "mpg", "webm", "m4v"]
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Transcribe video"):
            reset_transcript_state()

            original_name = os.path.splitext(uploaded_file.name)[0]
            file_base = safe_filename(original_name)

            with st.spinner("Loading Whisper model..."):
                model = load_whisper_model(model_name)

            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, uploaded_file.name)
                audio_path = os.path.join(temp_dir, f"{file_base}.mp3")

                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    with st.spinner("Extracting audio from video..."):
                        extract_audio_from_video(video_path, audio_path)

                    with st.spinner("Transcribing audio..."):
                        result = model.transcribe(audio_path)
                        transcript = result.get("text", "").strip()

                    if not transcript:
                        st.warning("No transcript text was produced.")
                    else:
                        st.success("Transcription complete.")
                        st.session_state.transcript = transcript
                        st.session_state.transcription_complete = True
                        st.session_state.source_label = "Uploaded video"
                        st.session_state.file_base = file_base
                        st.session_state.original_name = original_name

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    else:
        st.info("Upload a video file to get started.")

elif input_mode == "YouTube Link":
    youtube_url = st.text_input(
        "Paste a YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if st.button("Get YouTube captions"):
        reset_transcript_state()

        if not youtube_url.strip():
            st.warning("Please paste a YouTube URL.")
        elif not is_valid_youtube_url(youtube_url):
            st.warning("That does not look like a valid YouTube URL.")
        else:
            try:
                with st.spinner("Fetching YouTube captions..."):
                    transcript_text, fallback_title = fetch_youtube_captions(youtube_url)

                st.success("Captions loaded.")
                st.session_state.transcript = transcript_text
                st.session_state.transcription_complete = True
                st.session_state.source_label = "YouTube captions"
                st.session_state.file_base = safe_filename(fallback_title)
                st.session_state.original_name = fallback_title

            except Exception as e:
                st.error(f"Something went wrong: {e}")


# =========================
# TRANSCRIPT OUTPUT
# =========================
if st.session_state.transcription_complete and st.session_state.transcript:
    st.subheader("Transcript")
    st.caption(f"Source: {st.session_state.source_label}")

    edited_transcript = st.text_area(
        "Editable transcript",
        value=st.session_state.transcript,
        height=400
    )

    st.session_state.transcript = edited_transcript

    copy_to_clipboard_html(st.session_state.transcript)

    txt_bytes = st.session_state.transcript.encode("utf-8")
    pdf_bytes = make_pdf_bytes(
        title=f"Transcript - {st.session_state.original_name}",
        body_text=st.session_state.transcript
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download as TXT",
            data=txt_bytes,
            file_name=f"{st.session_state.file_base}_transcript.txt",
            mime="text/plain"
        )

    with col2:
        st.download_button(
            label="Download as PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.file_base}_transcript.pdf",
            mime="application/pdf"
        )