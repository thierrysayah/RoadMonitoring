# add total fixing cost to interface
# add option to add more boxes ?????
# add try catch for opening excel file
# add GPS data to infered images : done but time is not up to date



# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 18:49:31 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
Pothole Detection Dashboard (FreeSimpleGUI version)
Generates: CSV, Folium map (HTML), and Google Earth KML file.
"""

import os
import io
import base64
import pandas as pd
import FreeSimpleGUI as sg
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI conflict)
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import piexif
import folium
from pykml.factory import KML_ElementMaker as KML
from lxml import etree

import webbrowser


# ---------------------- Helper Functions ----------------------

def get_decimal_from_dms(dms, ref):
    degrees, minutes, seconds = dms
    decimal = degrees[0] / degrees[1] + minutes[0] / minutes[1] / 60 + seconds[0] / seconds[1] / 3600
    return -decimal if ref in ['S', 'W'] else decimal


def extract_gps_data(image_path):
    """Extract GPS data (lat, lon, alt) from image EXIF."""
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
    except:
        return ("n/a", "n/a", "n/a")


def plot_to_base64(df):
    """Convert a matplotlib plot to base64 for PySimpleGUI display."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df['image_name'], df['Fix Cost'], color='crimson', marker='o')
    ax.set_xlabel("Image")
    ax.set_ylabel("Fixing Cost ($)")
    ax.set_title("Fix Cost per Image")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read())


def convert_to_png_bytes(image_path, max_size=(400, 400)):
    """Convert image (any format) to PNG bytes for GUI preview."""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception as e:
        print(f"⚠️ Error converting image {image_path}: {e}")
        return None


# ---------------------- GUI Layout ----------------------

sg.theme("DarkBlue3")

left_col = [
    [sg.Text("Select Input Folder:")],
    [sg.InputText(key="-IN_FOLDER-"), sg.FolderBrowse()],
    [sg.Text("Select Output Folder:")],
    [sg.InputText(key="-OUT_FOLDER-"), sg.FolderBrowse()],

    [sg.HorizontalSeparator()],

    # [sg.Text("Image Size (pixels):", size=(20,1)), sg.InputText("16842816", key="-IMAGE_SIZE-", size=(15,1))],
    [sg.Text("Actual Ground Area (m²):", size=(20,1)), sg.InputText("20", key="-GROUND_AREA-", size=(15,1))],
    [sg.Text("Fixing Cost per m² ($):", size=(20,1)), sg.InputText("20", key="-COST_PER_M2-", size=(15,1))],

    [sg.HorizontalSeparator()],

    [sg.Button("Run Inference", size=(15,1), button_color=("white", "#007BFF"))],
    [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-PROG-')],
    [sg.Text("", key="-STATUS-", text_color="lightgreen")]
]

right_col = [
    [sg.Image(key="-GRAPH-", size=(600, 250))],
    [sg.Text("Processed Images:")],
    [sg.Listbox(values=[], size=(40, 10), key="-IMAGE_LIST-", enable_events=True),
     sg.Image(key="-IMAGE_PREVIEW-", size=(300, 200))]
]

layout = [
    [sg.Column(left_col, vertical_alignment="top"),
     sg.VSeperator(),
     sg.Column(right_col)]
]

window = sg.Window("Pothole Detection Dashboard", layout, resizable=True, finalize=True)

# ---------------------- Main Logic ----------------------

model = YOLO("G:/Startup/Site Seeing/road crack - pothole detection/pothole yolo11 oct 19.pt")

while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, "Exit"):
        break

    if event == "Run Inference":
        input_dir = values["-IN_FOLDER-"]
        output_dir = values["-OUT_FOLDER-"]
        if not input_dir or not output_dir:
            window["-STATUS-"].update("⚠️ Please select both folders", text_color="yellow")
            continue

        records = []
        try:
            # image_size = 1048576 #(1024*1024)
            total_ground_area = float(values["-GROUND_AREA-"])
            cost_per_m2 = float(values["-COST_PER_M2-"])
        except ValueError:
            window["-STATUS-"].update("⚠️ Invalid numeric input. Please check the values.", text_color="yellow")
            continue
        
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(image_files)

        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(input_dir, img_name)
            with Image.open(img_path) as img:
                width, height = img.size
                image_size = width * height
                print(image_size)
            
            results = model(img_path, conf=0.3,  imgsz=1024)
            save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_result.jpg")
            results[0].save(filename=save_path)
            
            try:
                # Load EXIF data from original
                original_img = Image.open(img_path)
                exif_data = original_img.info.get('exif')
                if exif_data:
                    # Reload saved (inferred) image and re-embed EXIF
                    inferred_img = Image.open(save_path)
                    inferred_img.save(save_path, exif=exif_data)
                    print(f"📍 GPS data preserved for: {img_name}")
                else:
                    print(f"⚠️ No EXIF found for: {img_name}")
            except Exception as e:
                print(f"❌ Failed to copy GPS data for {img_name}: {e}")
            

            lat, lon, alt = extract_gps_data(img_path)
            for result in results:
                boxes = result.boxes.xyxy
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(float, box[:4])
                        area = (x2 - x1) * (y2 - y1)
                        records.append({
                            "image_name": img_name,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "latitude": lat,
                            "longitude": lon,
                            "altitude": alt,
                            "area_pixels": area,
                            "image size":image_size
                        })

            del results
            window["-PROG-"].update(int(100 * (idx + 1) / total))

        # --- Data Aggregation ---
        df = pd.DataFrame(records)
        df['Total area per image'] = df.groupby('image_name')['area_pixels'].transform('sum')
        unique_df = df.groupby('image_name')[['latitude', 'longitude', 'altitude', 'area_pixels','image size']].agg({
            'latitude': 'first', 'longitude': 'first', 'altitude': 'first', 'area_pixels': 'sum', 'image size':'first'
        }).reset_index().rename(columns={'area_pixels': 'total pothole area'})
        unique_df['Fix Cost'] = (unique_df['total pothole area'] / unique_df['image size']) * total_ground_area * cost_per_m2
        unique_df['# of Potholes'] = df.groupby('image_name').size().values
        
        # --- Plot Graph ---
        graph_img = plot_to_base64(unique_df)
        window["-GRAPH-"].update(data=graph_img)

        # --- Populate Image List ---
        window["-IMAGE_LIST-"].update(values=image_files)

        # --- Save Outputs ---
        csv_path = os.path.join(output_dir, "potholes_summary.csv")
        unique_df.to_csv(csv_path, index=False)

        # --- Folium Map ---
        avg_lat = pd.to_numeric(df['latitude'], errors='coerce').mean()
        avg_lon = pd.to_numeric(df['longitude'], errors='coerce').mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=16, tiles="CartoDB positron")

       # === Helper to choose color scale based on cost ===
        def cost_color(cost):
           if cost < 2:
               return "green"
           elif cost < 5:
               return "orange"
           else:
               return "red"

       # === Add markers ===
        for _, row in unique_df.iterrows():
           try:
               lat, lon = float(row["latitude"]), float(row["longitude"])
           except:
               continue  # skip invalid coordinates

           img_path = os.path.join(input_dir, row["image_name"])
           if os.path.exists(img_path):
               # Encode image to base64 to embed in popup
               with open(img_path, "rb") as f:
                   encoded = base64.b64encode(f.read()).decode("utf-8")
               img_html = f'<img src="data:image/jpeg;base64,{encoded}" width="200">'
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
               radius=max(4, min(row["Fix Cost"], 20)),  # radius proportional to cost
               color=cost_color(row["Fix Cost"]),
               fill=True,
               fill_opacity=0.8,
               popup=folium.Popup(popup_html, max_width=250),
               tooltip=f"{row['image_name']} | ${row['Fix Cost']:.2f}"
           ).add_to(m)
           folium_path = os.path.join(output_dir, "pothole_map.html")
           m.save(folium_path)

        # --- KML File ---
        output_kml_path = os.path.join(output_dir, "pothole_map.kml")
        doc = KML.kml(KML.Document(KML.Name("Pothole Detection Results")))

        for _, row in unique_df.iterrows():
            try:
                lat, lon = float(row['latitude']), float(row['longitude'])
            except:
                continue

            img_path = os.path.join(input_dir, row['image_name'])
            if os.path.exists(img_path):
                img_tag = f'"file:///{img_path}">'
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

        with open(output_kml_path, "wb") as f:
            f.write(etree.tostring(doc, pretty_print=True))
        
        webbrowser.open(output_kml_path)
        webbrowser.open(folium_path)
        window["-STATUS-"].update(f"✅ Inference completed.\nResults saved to: {output_dir}", text_color="lightgreen")

    elif event == "-IMAGE_LIST-":
        selected = values["-IMAGE_LIST-"]
        if selected:
            img_name = selected[0]
            img_path = os.path.join(values["-OUT_FOLDER-"], f"{os.path.splitext(img_name)[0]}_result.jpg")
            if os.path.exists(img_path):
                img_bytes = convert_to_png_bytes(img_path)
                if img_bytes:
                    window["-IMAGE_PREVIEW-"].update(data=img_bytes)
                else:
                    window["-IMAGE_PREVIEW-"].update("⚠️ Unable to display image")
            else:
                window["-IMAGE_PREVIEW-"].update("No preview available")

window.close()
