# app.py
# Pothole Detection Dashboard — Streamlit Edition
# Runs YOLO inference on uploaded images, preserves EXIF GPS,
# outputs CSV, Folium HTML map (embedded), and KML download.

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Tuple, List

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import piexif
from ultralytics import YOLO
import folium
from pykml.factory import KML_ElementMaker as KML
from lxml import etree


# =========================
# ------- Utilities -------
# =========================

def get_decimal_from_dms(dms, ref):
    """Convert EXIF DMS to decimal degrees."""
    degrees, minutes, seconds = dms
    decimal = degrees[0] / degrees[1] + minutes[0] / minutes[1] / 60 + seconds[0] / seconds[1] / 3600
    return -decimal if ref in ['S', 'W'] else decimal

def extract_gps_data(image_path: str) -> Tuple[float, float, float] | Tuple[str, str, str]:
    """Extract (lat, lon, alt) from EXIF. Returns 'n/a' triplet if missing."""
    try:
        img = Image.open(image_path)
        exif_data = img.info.get('exif')
        if not exif_data:
            return ("n/a", "n/a", "n/a")
        exif_dict = piexif.load(exif_data)
        gps_data = exif_dict.get('GPS', {})
        if not gps_data:
            return ("n/a", "n/a", "n/a")
        lat = get_decimal_from_dms(
            gps_data[piexif.GPSIFD.GPSLatitude],
            gps_data[piexif.GPSIFD.GPSLatitudeRef].decode()
        )
        lon = get_decimal_from_dms(
            gps_data[piexif.GPSIFD.GPSLongitude],
            gps_data[piexif.GPSIFD.GPSLongitudeRef].decode()
        )
        alt = gps_data.get(piexif.GPSIFD.GPSAltitude)
        alt = alt[0] / alt[1] if alt else "n/a"
        return lat, lon, alt
    except Exception:
        return ("n/a", "n/a", "n/a")

def preserve_exif(src_path: str, dst_path: str) -> None:
    """Copy EXIF (including GPS) from src image to dst image if present."""
    try:
        original_img = Image.open(src_path)
        exif_data = original_img.info.get('exif')
        if exif_data:
            inferred_img = Image.open(dst_path)
            inferred_img.save(dst_path, exif=exif_data)
    except Exception:
        pass

def fig_cost_per_image(unique_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(unique_df['image_name'], unique_df['Fix Cost'], color='crimson', marker='o')
    ax.set_xlabel("Image")
    ax.set_ylabel("Fix Cost ($)")
    ax.set_title("Fix Cost per Image")
    ax.grid(alpha=0.2)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def embed_folium_map(unique_df: pd.DataFrame, originals_dir: Path) -> folium.Map:
    # center
    lat_mean = pd.to_numeric(unique_df['latitude'], errors='coerce').mean()
    lon_mean = pd.to_numeric(unique_df['longitude'], errors='coerce').mean()
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=16, tiles="CartoDB positron")

    def cost_color(cost):
        if cost < 2:
            return "green"
        elif cost < 5:
            return "orange"
        else:
            return "red"

    for _, row in unique_df.iterrows():
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
        except Exception:
            continue

        img_path = originals_dir / row["image_name"]
        if img_path.exists():
            with open(img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            img_html = f'<img src="data:image/jpeg;base64,{encoded}" width="220">'
        else:
            img_html = "<i>Image not found</i>"

        popup_html = f"""
        <div style="font-size:13px;">
            <b>{row['image_name']}</b><br>
            <b>Cost:</b> ${row['Fix Cost']:.2f}<br>
            <b>Altitude:</b> {row['altitude']} m<br>
            {img_html}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=max(4, min(float(row["Fix Cost"]), 20)),
            color=cost_color(float(row["Fix Cost"])),
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['image_name']} | ${row['Fix Cost']:.2f}"
        ).add_to(m)
    return m

def kml_bytes(unique_df: pd.DataFrame, originals_dir: Path) -> bytes:
    doc = KML.kml(KML.Document(KML.Name("Pothole Detection Results")))
    for _, row in unique_df.iterrows():
        try:
            lat, lon = float(row['latitude']), float(row['longitude'])
        except Exception:
            continue
        img_path = originals_dir / row['image_name']
        if img_path.exists():
            img_tag = f'"file:///{img_path.as_posix()}">'
        else:
            img_tag = "<i>Image not found</i>"

        placemark = KML.Placemark(
            KML.name(row['image_name']),
            KML.description(f"""
            <![CDATA[
            <b>Fix Cost:</b> ${row['Fix Cost']:.2f}<br>
            <b>Altitude:</b> {row['altitude']} m<br>
            <img style="max-width:500px;" src={img_tag}
            ]]>
            """),
            KML.Point(KML.coordinates(f"{lon},{lat},{row['altitude']}"))
        )
        doc.Document.append(placemark)
    return etree.tostring(doc, pretty_print=True)

@st.cache_resource  # keep model in memory across reruns
def load_model(model_path: str) -> YOLO:
    # Streamlit Cloud does not offer GPU; force CPU by default.
    return YOLO(model_path)

def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile], dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        out = dest_dir / f.name
        with open(out, "wb") as w:
            w.write(f.getbuffer())
        saved.append(out)
    return saved


# =========================
# --------- UI -----------
# =========================

st.set_page_config(page_title="Pothole Detection Dashboard", layout="wide")
st.title("🛣️ Pothole Detection Dashboard (Streamlit)")

# Sidebar controls
with st.sidebar:
    st.header("Parameters")
    model_path = st.text_input("Model (.pt) path", value="pothole yolo11 oct 19.pt")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.30, 0.01)
    imgsz = st.select_slider("Inference image size", options=[640, 768, 896, 1024, 1280], value=1024)
    total_ground_area = st.number_input("Actual ground area (m²) per image", value=20.0, min_value=0.0, step=1.0)
    cost_per_m2 = st.number_input("Fixing cost per m² ($)", value=20.0, min_value=0.0, step=1.0)
    st.caption("Ground area is the real-world area represented by a full image.")

st.markdown("#### Upload images")
uploads = st.file_uploader("JPEG/PNG images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

run = st.button("🚀 Run Inference", type="primary", disabled=(not uploads))

# Workspace
col_left, col_right = st.columns([1.1, 1])

if run:
    if not uploads:
        st.warning("Please upload at least one image.")
        st.stop()

    # Temp workspace
    tmp_root = Path(tempfile.mkdtemp(prefix="pothole_app_"))
    originals_dir = tmp_root / "originals"
    outputs_dir = tmp_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = save_uploaded_files(uploads, originals_dir)

    # Load model once
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model at `{model_path}`.\n{e}")
        st.stop()

    records = []
    progress = st.progress(0)
    status = st.empty()

    for i, img_path in enumerate(saved_paths, start=1):
        status.write(f"Processing {img_path.name} ({i}/{len(saved_paths)})…")

        # image size (px)
        with Image.open(img_path) as im:
            width, height = im.size
            image_size = width * height

        # Run YOLO
        results = model(str(img_path), conf=conf, imgsz=imgsz, device="cpu")
        out_path = outputs_dir / f"{img_path.stem}_result.jpg"
        results[0].save(filename=str(out_path))

        # Preserve EXIF GPS
        preserve_exif(str(img_path), str(out_path))

        # Extract GPS
        lat, lon, alt = extract_gps_data(str(img_path))

        # Record bboxes
        for res in results:
            boxes = res.boxes.xyxy
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(float, box[:4])
                    area = (x2 - x1) * (y2 - y1)
                    records.append({
                        "image_name": img_path.name,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "latitude": lat, "longitude": lon, "altitude": alt,
                        "area_pixels": area,
                        "image size": image_size,
                        "width": width, "height": height
                    })

        progress.progress(i / len(saved_paths))

    status.empty()

    # Build DataFrames
    df = pd.DataFrame(records)
    if df.empty:
        st.warning("No detections found in the uploaded images.")
        st.stop()

    df['Total area per image'] = df.groupby('image_name')['area_pixels'].transform('sum')

    agg = df.groupby('image_name')[['latitude','longitude','altitude','area_pixels','image size']] \
            .agg({'latitude':'first','longitude':'first','altitude':'first','area_pixels':'sum','image size':'first'}) \
            .reset_index().rename(columns={'area_pixels': 'total pothole area'})

    agg['# of Potholes'] = df.groupby('image_name').size().values
    agg['Fix Cost'] = (agg['total pothole area'] / agg['image size']) * total_ground_area * cost_per_m2

    # Split view: table + plot
    with col_left:
        st.subheader("📄 Summary (per image)")
        st.dataframe(agg, use_container_width=True, hide_index=True)

        # CSV download
        csv_bytes = agg.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="potholes_summary.csv", mime="text/csv")

        # KML download
        kml_data = kml_bytes(agg, originals_dir)
        st.download_button("🌍 Download KML (Google Earth)", data=kml_data,
                           file_name="pothole_map.kml", mime="application/vnd.google-earth.kml+xml")

    with col_right:
        st.subheader("📈 Cost per Image")
        st.pyplot(fig_cost_per_image(agg))

        st.subheader("🗺️ Map")
        fmap = embed_folium_map(agg, originals_dir)
        st.components.v1.html(fmap._repr_html_(), height=600, scrolling=False)

    # Image gallery
    st.subheader("🖼️ Processed Images")
    out_images = sorted(outputs_dir.glob("*_result.jpg"))
    cols = st.columns(3)
    for idx, p in enumerate(out_images):
        with cols[idx % 3]:
            st.image(str(p), caption=p.name, use_container_width=True)

else:
    st.info("Upload images and click **Run Inference** to begin.")
