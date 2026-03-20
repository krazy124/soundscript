import io
import os
import html
import tempfile

import streamlit as st
import whisper

from moviepy.video.io.VideoFileClip import VideoFileClip
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


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
    Extract audio from a video file and save it as MP3.
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

        # Wrap long lines manually
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


# =========================
# UI
# =========================
st.title("Video to Text Transcriber")
st.write("Upload a video, extract the audio, transcribe it with Whisper, and download the transcript.")

with st.expander("Settings", expanded=True):
    model_name = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Smaller models are faster. Larger models may be more accurate."
    )

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi", "mkv", "mpeg", "mpg", "webm", "m4v"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    original_name = os.path.splitext(uploaded_file.name)[0]
    file_base = safe_filename(original_name)

    if st.button("Transcribe video"):
        with st.spinner("Loading Whisper model..."):
            model = load_whisper_model(model_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, uploaded_file.name)
            audio_path = os.path.join(temp_dir, f"{file_base}.mp3")

            # Save uploaded video to disk
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

                    st.subheader("Transcript")
                    st.text_area(
                        "Editable transcript",
                        value=transcript,
                        height=400
                    )

                    copy_to_clipboard_html(transcript)

                    txt_bytes = transcript.encode("utf-8")
                    pdf_bytes = make_pdf_bytes(
                        title=f"Transcript - {original_name}",
                        body_text=transcript
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label="Download as TXT",
                            data=txt_bytes,
                            file_name=f"{file_base}_transcript.txt",
                            mime="text/plain"
                        )

                    with col2:
                        st.download_button(
                            label="Download as PDF",
                            data=pdf_bytes,
                            file_name=f"{file_base}_transcript.pdf",
                            mime="application/pdf"
                        )

            except Exception as e:
                st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a video file to get started.")
