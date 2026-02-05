import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from copy import deepcopy
import matplotlib.pyplot as plt
import io

# --- Core Image Processing Logic ---

def process_image_data(image_bytes, contact_width_mm, threshold_percent, tyre_name):
    try:
        threshold_limit = int(float(threshold_percent) * 2.55)

        # Open image from in-memory bytes and convert to grayscale
        temp = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Apply median filter to reduce noise
        blur = temp.filter(ImageFilter.MedianFilter(size=7))

        imx = np.array(blur)
        number_of_black_pix = np.sum(imx <= threshold_limit)
        number_of_white_pix = np.sum(imx > threshold_limit)
        sum_pix = number_of_white_pix + number_of_black_pix
        
        percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        # Adjust threshold automatically
        while percent_black <= 0.55:
            threshold_limit += 1
            number_of_black_pix = np.sum(imx <= threshold_limit)
            percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        while percent_black >= 0.8:
            threshold_limit -= 1
            number_of_black_pix = np.sum(imx <= threshold_limit)
            percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        # Convert image array to binary format
        imx3 = np.where(imx <= threshold_limit, 0, 255).astype(np.uint8)
        bnw_image = Image.fromarray(imx3)
        im = bnw_image.convert('1')
        na = np.array(im)

        # --- NEW MIDPOINT SELECTION LOGIC ---
        # Find all black pixels (where na is False/0)
        black_pixel_coords = np.argwhere(na == False)
        
        if black_pixel_coords.size == 0:
            st.error("No footprint detected. Please adjust the threshold or check your image.")
            return None

        # Determine the footprint boundaries based on actual contact length
        min_row, min_col = black_pixel_coords.min(axis=0)
        max_row, max_col = black_pixel_coords.max(axis=0)

        # Calculate the Midpoints based on the contact length/width boundaries
        M = (min_row + max_row) // 2
        N = (min_col + max_col) // 2

        # Define the four quadrants using the footprint midpoint
        # We slice the original array 'na' into 4 segments
        Cells_Deepcopy = [
            na[min_row:M, min_col:N], # Top Left
            na[min_row:M, N:max_col], # Top Right
            na[M:max_row, min_col:N], # Bottom Left
            na[M:max_row, N:max_col]  # Bottom Right
        ]

        # Safety check for quadrant size
        for i, cell in enumerate(Cells_Deepcopy):
            if cell.shape[0] == 0 or cell.shape[1] == 0:
                st.error(f"Slicing failed: Quadrant {i+1} has no data. Check image alignment.")
                return None

        # --- End of New Slicing Logic ---

        # Define the painter function to fill vertical gaps
        def painter(array):
            a1, b1 = array.shape
            for j in range(b1):
                pollute = 2
                for i in range(a1):
                    if not array[i][j]: # if pixel is black
                        pollute = 1
                    elif array[i][j] and pollute != 2: # if pixel is white and we've seen a black pixel
                        array[i][j] = False # fill the gap
            return array

        # Process quadrants
        arr2D_TL_Act_Fill = deepcopy(Cells_Deepcopy[0])
        arr2D_TR_Act_Fill = deepcopy(Cells_Deepcopy[1])
        arr2D_BL_Act_Fill = deepcopy(Cells_Deepcopy[2])
        arr2D_BR_Act_Fill = deepcopy(Cells_Deepcopy[3])

        arr2D_TL_Flp_Fill = deepcopy(np.flipud(arr2D_TL_Act_Fill))
        arr2D_TR_Flp_Fill = deepcopy(np.flipud(arr2D_TR_Act_Fill))
        arr2D_BL_Flp_Fill = deepcopy(np.flipud(arr2D_BL_Act_Fill))
        arr2D_BR_Flp_Fill = deepcopy(np.flipud(arr2D_BR_Act_Fill))
        
        array_listT = [arr2D_TL_Flp_Fill, arr2D_TR_Flp_Fill, arr2D_TL_Act_Fill, arr2D_TR_Act_Fill]
        array_listB = [arr2D_BL_Flp_Fill, arr2D_BR_Flp_Fill, arr2D_BL_Act_Fill, arr2D_BR_Act_Fill]
        
        for item in array_listT:
             item = painter(item)
        
        for item in array_listB:
             item = painter(item)
             item = np.flipud(item)

        array_listF =  [np.logical_or(arr2D_TL_Flp_Fill, np.flipud(arr2D_TL_Act_Fill)),
                        np.logical_or(arr2D_TR_Flp_Fill, np.flipud(arr2D_TR_Act_Fill)),
                        np.logical_or(arr2D_BL_Flp_Fill, np.flipud(arr2D_BL_Act_Fill)),
                        np.logical_or(arr2D_BR_Flp_Fill, np.flipud(arr2D_BR_Act_Fill))]    
        
        arrT = np.hstack((array_listF[0], array_listF[1]))
        arrT = np.flipud(arrT)
        arrB = np.hstack((array_listF[2], array_listF[3]))
        arrB = np.flipud(arrB)
        arr2D3 = np.vstack((arrT, arrB)) 

        # --- Calculations ---
        contact_width_pixels = max(np.sum(arr2D3 == 0, axis=0))
        contact_length_pixels = max(np.sum(arr2D3 == 0, axis=1))
        pixels_per_mm = float(contact_width_pixels) / float(contact_width_mm)
        contact_length_mm = round(float(contact_length_pixels) / float(pixels_per_mm), 1)

        black_initial = np.count_nonzero(np.array(im) == 0)
        black_filled = np.count_nonzero(arr2D3 == 0)

        net_area_mm2 = (black_initial / pixels_per_mm**2)
        net_area_in2 = round(net_area_mm2 * 0.00155, 2)
        
        gross_area_mm2 = (black_filled / pixels_per_mm**2)
        gross_area_in2 = round(gross_area_mm2 * 0.00155, 2)

        # --- Plotting ---
        height, width = temp.size
        font_number = min(int(height / 30), 13)
        result_text_offset_y= -(0.02*font_number)
        result_text_offset_x = 0.5
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout(pad=2.0)

        title = f"{tyre_name} | Footprint Analysis" if tyre_name else "Footprint Analysis"
        fig.suptitle(title, fontsize=font_number + 4, fontweight='bold', color='blue')

        ax[0].imshow(temp, cmap='gray')
        ax[0].set_title("Original Image", fontsize=font_number, fontweight='bold', color='green')
        ax[0].axis("off")
        ax[0].text(result_text_offset_x,result_text_offset_y, f"Contact Width: {contact_width_mm} mm", transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.9), color='green')
        ax[0].text(result_text_offset_x,result_text_offset_y-0.2, f"Contact Length: {contact_length_mm} mm", transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.9))

        ax[1].imshow(imx3, cmap='gray')
        ax[1].set_title("Processed Image", fontsize=font_number, fontweight='bold')
        ax[1].axis("off")
        ax[1].text(result_text_offset_x,result_text_offset_y, f"Net Area: {net_area_in2} sq.in", transform=ax[1].transAxes, ha='center', fontsize=int(font_number*0.9))

        ax[2].imshow(arr2D3, cmap='gray')
        ax[2].set_title("Filled Image", fontsize=font_number, fontweight='bold')
        ax[2].axis("off")
        ax[2].text(result_text_offset_x,result_text_offset_y, f"Gross Area: {gross_area_in2} sq.in", transform=ax[2].transAxes, ha='center', fontsize=int(font_number*0.9))

        return fig

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None

# --- Streamlit GUI ---
st.set_page_config(layout="wide", page_title="Tyre Contact Area Analysis")
st.title("Tyre Contact Area Generator")

if 'final_figure' not in st.session_state:
    st.session_state.final_figure = None

with st.sidebar:
    st.header("⚙️ Parameters")
    uploaded_file = st.file_uploader("Select Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    contact_width = st.number_input("Contact Width (mm)", min_value=1.0, value=150.0, step=1.0)
    threshold = st.slider("Black Recognition % (0-100)", min_value=0, max_value=100, value=60)
    tyre_name_input = st.text_input("Tyre & OST Name (Optional)")
    process_button = st.button("Process Image", type="primary")

if process_button:
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        with st.spinner('Analyzing image...'):
            st.session_state.final_figure = process_image_data(image_bytes, contact_width, threshold, tyre_name_input)
    else:
        st.warning("Please upload an image file first.")

if st.session_state.final_figure is not None:
    st.subheader("Analysis Results")
    st.pyplot(st.session_state.final_figure)
