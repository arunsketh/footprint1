import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from copy import deepcopy
import matplotlib.pyplot as plt
import io

# --- Core Image Processing Logic (Adapted from your Tkinter script) ---
# Note: Functions are now independent and don't rely on a class structure.

def process_image_data(image_bytes, contact_width_mm, threshold_percent, tyre_name):
    """
    This function contains the main image processing pipeline from your script.
    It takes image data and parameters, and returns the final figure to display.
    """
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
        # Handle division by zero if sum_pix is 0
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

        # Convert image array to binary format (0 for black, 255 for white)
        imx3 = np.where(imx <= threshold_limit, 0, 255).astype(np.uint8)
        bnw_image = Image.fromarray(imx3)

        # --- Gap Filling Logic (painter function and array manipulation) ---
        im = bnw_image.convert('1')
        
        na = np.array(im)
        Pixel_X, Pixel_Y = na.shape
        Cut_offset = 1
        
        # Boundary check for mid_point calculation
        if Pixel_X // 2 >= Pixel_X or Pixel_Y // 2 >= Pixel_Y:
            st.error("Image dimensions are too small for processing.")
            return None
            
        mid_point = na[Pixel_X // 2, Pixel_Y // 2]
        while mid_point:
            Cut_offset += 1
            if (Pixel_X // 2 + Cut_offset) >= Pixel_X: # Boundary check
                st.warning("Could not determine mid-point offset. Results may be affected.")
                break
            mid_point = na[Pixel_X // 2 + Cut_offset, Pixel_Y // 2]

        M = (Pixel_X // 2) + Cut_offset + 5
        N = Pixel_Y // 2

        if M <= 0 or N <= 0:
            st.error("Calculated quadrant dimensions are invalid. Please check image integrity.")
            return None

        Cells = [na[x:x + M, y:y + N] for x in range(0, na.shape[0], M) for y in range(0, na.shape[1], N)]
        if len(Cells) < 4:
            st.error("Image processing failed: Could not split the image into four quadrants. Try a different image or adjust parameters.")
            return None

        Cells_Deepcopy = deepcopy([cell for cell in Cells if cell.shape[0] > 1 and cell.shape[1] > 1])
        if len(Cells_Deepcopy) < 4:
            st.error("Image processing failed after filtering quadrants. Image may be too small or irregular.")
            return None

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

        # *** CORRECTED LOGIC TO REPLICATE ORIGINAL SCRIPT'S INTENT ***
        # Create copies for processing
        arr2D_TL_Act_Fill = deepcopy(Cells_Deepcopy[0])
        arr2D_TR_Act_Fill = deepcopy(Cells_Deepcopy[1])
        arr2D_BL_Act_Fill = deepcopy(Cells_Deepcopy[2])
        arr2D_BR_Act_Fill = deepcopy(Cells_Deepcopy[3])

        arr2D_TL_Flp_Fill = deepcopy(np.flipud(arr2D_TL_Act_Fill))
        arr2D_TR_Flp_Fill = deepcopy(np.flipud(arr2D_TR_Act_Fill))
        arr2D_BL_Flp_Fill = deepcopy(np.flipud(arr2D_BL_Act_Fill))
        arr2D_BR_Flp_Fill = deepcopy(np.flipud(arr2D_BR_Act_Fill))
        
        # Process top and bottom cells using painter function
        array_listT = [arr2D_TL_Flp_Fill, arr2D_TR_Flp_Fill, arr2D_TL_Act_Fill, arr2D_TR_Act_Fill]
        array_listB = [arr2D_BL_Flp_Fill, arr2D_BR_Flp_Fill, arr2D_BL_Act_Fill, arr2D_BR_Act_Fill]
        
        for item in array_listT:
             item = painter(item)
        
        for item in array_listB:
             item = painter(item)
             item = np.flipud(item)

        # Combine filled images using logical OR
        array_listF =  [np.logical_or(arr2D_TL_Flp_Fill, np.flipud(arr2D_TL_Act_Fill)),
                        np.logical_or(arr2D_TR_Flp_Fill, np.flipud(arr2D_TR_Act_Fill)),
                        np.logical_or(arr2D_BL_Flp_Fill, np.flipud(arr2D_BL_Act_Fill)),
                        np.logical_or(arr2D_BR_Flp_Fill, np.flipud(arr2D_BR_Act_Fill))]    
        
        # Recombine the fully processed quadrants into the final image
        arrT = np.hstack((array_listF[0], array_listF[1]))
        arrT = np.flipud(arrT)
        arrB = np.hstack((array_listF[2], array_listF[3]))
        arrB = np.flipud(arrB)
        arr2D3 = np.vstack((arrT, arrB)) # This is the final filled image array

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

        # Original Image
        ax[0].imshow(temp, cmap='gray')
        ax[0].set_title("Original Image", fontsize=font_number, fontweight='bold', color='green')
        ax[0].axis("off")
        ax[0].text(result_text_offset_x,result_text_offset_y, f"Contact Width: {contact_width_mm} mm", transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.9), color='green')
        ax[0].text(result_text_offset_x,result_text_offset_y-0.2, f"Contact Length: {contact_length_mm} mm", transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.9))

        # Processed (B&W) Image
        ax[1].imshow(imx3, cmap='gray')
        ax[1].set_title("Processed Image", fontsize=font_number, fontweight='bold')
        ax[1].axis("off")
        ax[1].text(result_text_offset_x,result_text_offset_y, f"Net Area: {net_area_in2} sq.in", transform=ax[1].transAxes, ha='center', fontsize=int(font_number*0.9))

        # Filled Image
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

# Initialize session state to hold the results
if 'final_figure' not in st.session_state:
    st.session_state.final_figure = None

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Parameters")
    
    uploaded_file = st.file_uploader("Select Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    contact_width = st.number_input("Contact Width (mm)", min_value=1.0, value=150.0, step=1.0)
    threshold = st.slider("Black Recognition % (0-100)", min_value=0, max_value=100, value=60)
    tyre_name_input = st.text_input("Tyre & OST Name (Optional)")
    process_button = st.button("Process Image", type="primary")

# --- Main App Logic ---
if process_button:
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        with st.spinner('Analyzing image...'):
            st.session_state.final_figure = process_image_data(image_bytes, contact_width, threshold, tyre_name_input)
    else:
        st.warning("Please upload an image file first.")

# Display the resulting figure if it exists in the session state
if st.session_state.final_figure is not None:
    st.subheader("Analysis Results")
    st.pyplot(st.session_state.final_figure)
