import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from copy import deepcopy
import matplotlib.pyplot as plt
import io

# --- Core Image Processing Logic ---

def process_image_data(image_bytes, contact_width_mm, threshold_percent, tyre_name):
    """
    Main pipeline: Grayscale -> Noise Filter -> Threshold -> Gap Filling -> Calculation.
    """
    try:
        # 1. PRE-PROCESSING & NOISE REDUCTION
        # Convert the 0-100 scale from the slider to a 0-255 grayscale integer
        threshold_limit = int(float(threshold_percent) * 2.55)

        # Load image from the byte stream uploaded by the user
        # .convert('L') turns the image into 8-bit grayscale (0=Black, 255=White)
        temp = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Apply a Median Filter (size 7x7) to "smooth" the image
        # This removes small dots/scratches without blurring the sharp edges of the tyre print
        blur = temp.filter(ImageFilter.MedianFilter(size=7))

        # Convert the filtered image into a NumPy array (a grid of numbers)
        imx = np.array(blur)
        
        # Initial count of pixels to determine if the image is mostly light or dark
        number_of_black_pix = np.sum(imx <= threshold_limit)
        number_of_white_pix = np.sum(imx > threshold_limit)
        sum_pix = number_of_white_pix + number_of_black_pix
        percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        # 2. AUTO-THRESHOLD CALIBRATION
        # This logic ensures the image isn't "washed out" or "totally black."
        # It forces the black pixel percentage to stay within a reasonable range (55%-80%)
        while percent_black <= 0.55:
            threshold_limit += 1 # Make it easier for pixels to be counted as black
            number_of_black_pix = np.sum(imx <= threshold_limit)
            percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        while percent_black >= 0.8:
            threshold_limit -= 1 # Make it harder for pixels to be counted as black
            number_of_black_pix = np.sum(imx <= threshold_limit)
            percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        # 3. BINARIZATION
        # Create a "Binary" image: strictly 0 (Black/Tyre) or 255 (White/Background)
        imx3 = np.where(imx <= threshold_limit, 0, 255).astype(np.uint8)
        bnw_image = Image.fromarray(imx3)

        # 4. IMAGE SEGMENTATION (Splitting into 4 parts)
        # We split the image to process each corner starting from the center outward.
        im = bnw_image.convert('1') # 1-bit pixels (True/False)
        na = np.array(im)
        Pixel_X, Pixel_Y = na.shape
        
        # Find the center point and adjust if the exact center pixel is empty
        Cut_offset = 1
        if Pixel_X // 2 >= Pixel_X or Pixel_Y // 2 >= Pixel_Y:
            st.error("Image dimensions are too small for processing.")
            return None
            
        mid_point = na[Pixel_X // 2, Pixel_Y // 2]
        while mid_point: # Move offset until we find the edge of a tread block
            Cut_offset += 1
            if (Pixel_X // 2 + Cut_offset) >= Pixel_X:
                st.warning("Could not determine mid-point offset.")
                break
            mid_point = na[Pixel_X // 2 + Cut_offset, Pixel_Y // 2]

        # Calculate dimensions for the four quadrants
        M = (Pixel_X // 2) + Cut_offset + 5
        N = Pixel_Y // 2

        if M <= 0 or N <= 0:
            st.error("Invalid dimensions calculated.")
            return None

        # Slice the array into quadrants
        Cells = [na[x:x + M, y:y + N] for x in range(0, na.shape[0], M) for y in range(0, na.shape[1], N)]
        if len(Cells) < 4:
            st.error("Failed to split image into quadrants.")
            return None

        # Filter out invalid or tiny slices
        Cells_Deepcopy = deepcopy([cell for cell in Cells if cell.shape[0] > 1 and cell.shape[1] > 1])
        if len(Cells_Deepcopy) < 4:
            st.error("Quadrants too small or irregular.")
            return None

        # 5. GAP FILLING (PAINTER) LOGIC
        # This function identifies where the tyre 'starts' and fills in all white 
        # space inside the footprint (tread grooves) to calculate Gross Area.
        def painter(array):
            a1, b1 = array.shape
            for j in range(b1): # Iterate through columns
                pollute = 2 # State 2: Background
                for i in range(a1): # Iterate through rows
                    if not array[i][j]: # If pixel is Black (Tyre contact)
                        pollute = 1 # Change state: We have entered the footprint
                    elif array[i][j] and pollute != 2: 
                        # If pixel is White BUT we are already inside the footprint state
                        array[i][j] = False # Paint the white pixel Black
            return array

        # 6. QUADRANT RECONSTRUCTION
        # Create copies of quadrants: TL (Top Left), TR (Top Right), BL, BR
        arr2D_TL_Act_Fill = deepcopy(Cells_Deepcopy[0])
        arr2D_TR_Act_Fill = deepcopy(Cells_Deepcopy[1])
        arr2D_BL_Act_Fill = deepcopy(Cells_Deepcopy[2])
        arr2D_BR_Act_Fill = deepcopy(Cells_Deepcopy[3])

        # Create flipped versions so the painter always works from "inside" to "outside"
        arr2D_TL_Flp_Fill = deepcopy(np.flipud(arr2D_TL_Act_Fill))
        arr2D_TR_Flp_Fill = deepcopy(np.flipud(arr2D_TR_Act_Fill))
        arr2D_BL_Flp_Fill = deepcopy(np.flipud(arr2D_BL_Act_Fill))
        arr2D_BR_Flp_Fill = deepcopy(np.flipud(arr2D_BR_Act_Fill))
        
        # Apply the filling algorithm to top and bottom groups
        array_listT = [arr2D_TL_Flp_Fill, arr2D_TR_Flp_Fill, arr2D_TL_Act_Fill, arr2D_TR_Act_Fill]
        array_listB = [arr2D_BL_Flp_Fill, arr2D_BR_Flp_Fill, arr2D_BL_Act_Fill, arr2D_BR_Act_Fill]
        
        for item in array_listT:
             item = painter(item)
        
        for item in array_listB:
             item = painter(item)
             item = np.flipud(item) # Flip back to original orientation

        # Combine the actual and flipped versions to ensure all internal gaps are closed
        array_listF =  [np.logical_or(arr2D_TL_Flp_Fill, np.flipud(arr2D_TL_Act_Fill)),
                        np.logical_or(arr2D_TR_Flp_Fill, np.flipud(arr2D_TR_Act_Fill)),
                        np.logical_or(arr2D_BL_Flp_Fill, np.flipud(arr2D_BL_Act_Fill)),
                        np.logical_or(arr2D_BR_Flp_Fill, np.flipud(arr2D_BR_Act_Fill))]    
        
        # Stitch the quadrants back together into one large 2D image
        arrT = np.hstack((array_listF[0], array_listF[1]))
        arrT = np.flipud(arrT)
        arrB = np.hstack((array_listF[2], array_listF[3]))
        arrB = np.flipud(arrB)
        arr2D3 = np.vstack((arrT, arrB)) # Final "Gross Area" image array

        # 7. SCALING & CALCULATIONS
        # Find the max width/length in pixels
        contact_width_pixels = max(np.sum(arr2D3 == 0, axis=0))
        contact_length_pixels = max(np.sum(arr2D3 == 0, axis=1))
        
        # Determine how many pixels represent 1 mm based on user input
        pixels_per_mm = float(contact_width_pixels) / float(contact_width_mm)
        contact_length_mm = round(float(contact_length_pixels) / float(pixels_per_mm), 1)

        # Count black pixels in the original binary vs the filled version
        black_initial = np.count_nonzero(np.array(im) == 0)
        black_filled = np.count_nonzero(arr2D3 == 0)

        # Area = (Pixel Count) / (Pixels per mm)^2
        # Convert mm^2 to square inches (0.00155 multiplier)
        net_area_mm2 = (black_initial / pixels_per_mm**2)
        net_area_in2 = round(net_area_mm2 * 0.00155, 2)
        
        gross_area_mm2 = (black_filled / pixels_per_mm**2)
        gross_area_in2 = round(gross_area_mm2 * 0.00155, 2)

        # 8. VISUALIZATION (Matplotlib)
        height, width = temp.size
        font_number = min(int(height / 30), 13) # Scale text based on image size
        result_text_offset_y= -(0.02*font_number)
        result_text_offset_x = 0.5
        
        # Create a side-by-side comparison plot
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout(pad=2.0)

        title = f"{tyre_name} | Footprint Analysis" if tyre_name else "Footprint Analysis"
        fig.suptitle(title, fontsize=font_number + 4, fontweight='bold', color='blue')

        # Subplot 1: Original Scan
        ax[0].imshow(temp, cmap='gray')
        ax[0].set_title("Original Image", fontsize=font_number, fontweight='bold', color='green')
        ax[0].axis("off")
        ax[0].text(result_text_offset_x,result_text_offset_y, f"Contact Width: {contact_width_mm} mm", transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.9), color='green')
        ax[0].text(result_text_offset_x,result_text_offset_y-0.2, f"Contact Length: {contact_length_mm} mm", transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.9))

        # Subplot 2: Processed Binary (Net Area)
        ax[1].imshow(imx3, cmap='gray')
        ax[1].set_title("Processed Image", fontsize=font_number, fontweight='bold')
        ax[1].axis("off")
        ax[1].text(result_text_offset_x,result_text_offset_y, f"Net Area: {net_area_in2} sq.in", transform=ax[1].transAxes, ha='center', fontsize=int(font_number*0.9))

        # Subplot 3: Filled Reconstruction (Gross Area)
        ax[2].imshow(arr2D3, cmap='gray')
        ax[2].set_title("Filled Image", fontsize=font_number, fontweight='bold')
        ax[2].axis("off")
        ax[2].text(result_text_offset_x,result_text_offset_y, f"Gross Area: {gross_area_in2} sq.in", transform=ax[2].transAxes, ha='center', fontsize=int(font_number*0.9))

        return fig

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None

# --- Streamlit Web Interface Configuration ---
st.set_page_config(layout="wide", page_title="Tyre Contact Area Analysis")
st.title("Tyre Contact Area Generator")

# Use session state to keep the image visible even if the user clicks other buttons
if 'final_figure' not in st.session_state:
    st.session_state.final_figure = None

# Create a Sidebar for settings
with st.sidebar:
    st.header("⚙️ Parameters")
    
    # Input 1: File Upload
    uploaded_file = st.file_uploader("Select Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    # Input 2: Physical Width (Needed for scaling pixels to mm)
    contact_width = st.number_input("Contact Width (mm)", min_value=1.0, value=150.0, step=1.0)
    
    # Input 3: Threshold sensitivity
    threshold = st.slider("Black Recognition % (0-100)", min_value=0, max_value=100, value=60)
    
    # Input 4: Labeling
    tyre_name_input = st.text_input("Tyre & OST Name (Optional)")
    
    # Input 5: Execution Button
    process_button = st.button("Process Image", type="primary")

# --- Main App Execution Logic ---
if process_button:
    if uploaded_file is not None:
        # Read the file content into memory
        image_bytes = uploaded_file.getvalue()
        with st.spinner('Analyzing image...'):
            # Run the heavy math and store the resulting plot in session state
            st.session_state.final_figure = process_image_data(image_bytes, contact_width, threshold, tyre_name_input)
    else:
        st.warning("Please upload an image file first.")

# If a figure was successfully generated, display it in the main area
if st.session_state.final_figure is not None:
    st.subheader("Analysis Results")
    st.pyplot(st.session_state.final_figure)
