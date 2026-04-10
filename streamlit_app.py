import streamlit as st
import image_compressor
from pathlib import Path
import io
import contextlib
import tempfile
import zipfile
import os

st.set_page_config(page_title="Batch Image Compressor", page_icon="🖼️", layout="centered")

st.title("🖼️ Batch Image Compressor (Web Version)")
st.markdown("Process images directly in the browser — upload, compress, and download!")

# Load defaults
default_cfg = image_compressor.DEFAULT

# Layout
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    output_format = st.selectbox("Output Format", ["webp"], index=0)
    quality = st.slider("Target Quality (1-100)", min_value=1, max_value=100, value=default_cfg["quality"])

with col2:
    min_size = st.number_input("Min Size (KB)", min_value=1, value=50)
    max_size = st.number_input("Max Size (KB)", min_value=1, value=200)
    threads = st.number_input("Threads (Workers)", min_value=1, max_value=32, value=default_cfg["threads"])

# Restrict to PNG and JPEG
image_compressor.SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

st.markdown("---")

uploaded_files = st.file_uploader("Upload Image Files (JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("🚀 Start Compression", width="stretch"):
    if not uploaded_files:
        st.warning("Please upload at least one image to begin.")
    else:
        with st.spinner("Compressing images... Please wait."):
            # Create temporary directories for input and output
            with tempfile.TemporaryDirectory() as temp_in, tempfile.TemporaryDirectory() as temp_out:
                
                # Save uploaded files temporarily
                for uploaded_file in uploaded_files:
                    file_path = Path(temp_in) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Build config
                cfg = default_cfg.copy()
                cfg["input_folder"] = temp_in
                cfg["output_folder"] = temp_out
                cfg["output_format"] = output_format
                cfg["quality"] = int(quality)
                cfg["threads"] = int(threads)
                cfg["max_size_kb"] = int(max_size)
                cfg["min_size_kb"] = int(min_size)
                
                # Set up UI elements for progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    progress_frac = current / total
                    progress_bar.progress(progress_frac)
                    status_text.info(f"⏳ Processed {current} of {total} images... ({(total - current)} remaining)")

                # Capture the console output
                stdout_stream = io.StringIO()
                with contextlib.redirect_stdout(stdout_stream):
                    try:
                        file_results = image_compressor.run(cfg, progress_callback=update_progress)
                        success = True
                        error_msg = ""
                    except Exception as e:
                        success = False
                        error_msg = str(e)
                
                if success:
                    status_text.empty()
                    st.success(f"✅ Compression finished successfully! Processed {len(uploaded_files)} images.")
                    
                    # Package the compressed images into a single ZIP file
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for root, _, files in os.walk(temp_out):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_out)
                                zip_file.write(file_path, arcname)
                    
                    st.download_button(
                        label="📥 Download Compressed Images (.zip)",
                        data=zip_buffer.getvalue(),
                        file_name="compressed_images.zip",
                        mime="application/zip",
                        width="stretch"
                    )

                    if file_results:
                        st.subheader("Results File by File")
                        st.dataframe(file_results)
                    
                    with st.expander("View Execution Logs & Summary", expanded=False):
                        st.text(stdout_stream.getvalue())
                else:
                    st.error(f"❌ An error occurred during compression:\n{error_msg}")
