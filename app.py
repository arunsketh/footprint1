import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from copy import deepcopy
import matplotlib.pyplot as plt
import io

# --- Core Image Processing Logic ---

def process_image_data(image_bytes, contact_width_mm, threshold_percent, tyre_name):
    """
    Takes image data and parameters, processes the footprint, and returns the figure.
    """
    try:
        threshold_limit = int(float(threshold_percent) * 2.55)

        # Open image and convert to grayscale
        temp = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Apply median filter to reduce noise
        blur = temp.filter(ImageFilter.MedianFilter(size=7))

        imx = np.array(blur)
        number_of_black_pix = np.sum(imx <= threshold_limit)
        number_of_white_pix = np.sum(imx > threshold_limit)
        sum_pix = number_of_white_pix + number_of_black_pix
        
        # Handle division by zero
        percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        # Adjust threshold automatically
        while percent_black <= 0.55 and threshold_limit < 255:
            threshold_limit += 1
            number_of_black_pix = np.sum(imx <= threshold_limit)
            percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        while percent_black >= 0.8 and threshold_limit > 0:
            threshold_limit -= 1
            number_of_black_pix = np.sum(imx <= threshold_limit)
            percent_black = (number_of_black_pix / sum_pix) if sum_pix > 0 else 0

        # Convert to binary format
        imx3 = np.where(imx <= threshold_limit, 0, 255).astype(np.uint8)
        bnw_image = Image.fromarray(imx3)

        im = bnw_image.convert('1')
        na = np.array(im)
        Pixel_X, Pixel_Y = na.shape

        black_pixels_per_row = np.sum(na == 0, axis=1)
        max_black_row_index = np.argmax(black_pixels_per_row)

        M = max_black_row_index
        N = Pixel_Y // 2

        # Safety check for bounds
        M = max(1, min(M, Pixel_X - 1))
        N = max(1, min(N, Pixel_Y - 1))

        # Explicit Slicing
        arr2D_TL_Act_Fill = deepcopy(na[0:M, 0:N])
        arr2D_TR_Act_Fill = deepcopy(na[0:M, N:])
        arr2D_BL_Act_Fill = deepcopy(na[M:, 0:N])
        arr2D_BR_Act_Fill = deepcopy(na[M:, N:])

        def painter(array):
            a1, b1 = array.shape
            for j in range(b1):
                pollute = 2
                for i in range(a1):
                    if not array[i][j]: 
                        pollute = 1
                    elif array[i][j] and pollute != 2:
                        array[i][j] = False
            return array
        
        # Process quadrants
        array_listT = [deepcopy(np.flipud(arr2D_TL_Act_Fill)), deepcopy(np.flipud(arr2D_TR_Act_Fill)), 
                       arr2D_TL_Act_Fill, arr2D_TR_Act_Fill]
        array_listB = [deepcopy(np.flipud(arr2D_BL_Act_Fill)), deepcopy(np.flipud(arr2D_BR_Act_Fill)), 
                       arr2D_BL_Act_Fill, arr2D_BR_Act_Fill]
        
        for item in array_listT:
             item = painter(item)
        
        for item in array_listB:
             item = painter(item)
             item = np.flipud(item)

        array_listF = [np.logical_or(array_listT[0], np.flipud(array_listT[2])),
                       np.logical_or(array_listT[1], np.flipud(array_listT[3])),
                       np.logical_or(array_listB[0], np.flipud(array_listB[2])),
                       np.logical_or(array_listB[1], np.flipud(array_listB[3]))]     
        
        arrT = np.flipud(np.hstack((array_listF[0], array_listF[1])))
        arrB = np.flipud(np.hstack((array_listF[2], array_listF[3])))
        arr2D3 = np.vstack((arrT, arrB))

        # Calculations
        contact_width_pixels = max(np.sum(arr2D3 == 0, axis=0))
        contact_length_pixels = max(np.sum(arr2D3 == 0, axis=1))
        pixels_per_mm = float(contact_width_pixels) / float(contact_width_mm)
        contact_length_mm = round(float(contact_length_pixels) / float(pixels_per_mm), 1)

        black_initial = np.count_nonzero(np.array(im) == 0)
        black_filled = np.count_nonzero(arr2D3 == 0)

        net_area_in2 = round((black_initial / pixels_per_mm**2) * 0.00155, 2)
        gross_area_in2 = round((black_filled / pixels_per_mm**2) * 0.00155, 2)

        # Plotting
        height, width = temp.size
        font_number = min(int(height / 30), 13)
        result_text_offset_y = -(0.02 * font_number)
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout(pad=2.0)
        title = f"{tyre_name} | Footprint Analysis" if tyre_name else "Footprint Analysis"
        fig.suptitle(title, fontsize=font_number + 4, fontweight='bold', color='blue')

        ax[0].imshow(temp, cmap='gray')
        ax[0].set_title("Original Image", fontsize=font_number, fontweight='bold', color='green')
        ax[0].axis("off")
        ax[0].text(0.5, result_text_offset_y, f"Width: {contact_width_mm}mm | Length: {contact_length_mm}mm", 
                   transform=ax[0].transAxes, ha='center', fontsize=int(font_number*0.8))

        ax[1].imshow(imx3, cmap='gray')
        ax[1].set_title("Processed Image", fontsize=font_number, fontweight='bold')
        ax[1].axis("off")
        ax[1].text(0.5, result_text_offset_y, f"Net Area: {net_area_in2} sq.in", 
                   transform=ax[1].transAxes, ha='center', fontsize=int(font_number*0.8))

        ax[2].imshow(arr2D3, cmap='gray')
        ax[2].set_title("Filled Image", fontsize=font_number, fontweight='bold')
        ax[2].axis("off")
        ax[2].text(0.5, result_text_offset_y, f"Gross Area: {gross_area_in2} sq.in", 
                   transform=ax[2].transAxes, ha='center', fontsize=int(font_number*0.8))

        return fig

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def save_plot_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Tyre Contact Area Analysis")
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'final_figure' not in st.session_state:
    st.session_state.final_figure = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Parameters")
    uploaded_file = st.file_uploader("Select Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    contact_width = st.number_input("Contact Width (mm)", min_value=1.0, value=150.0, step=1.0)
    threshold = st.slider("Black Recognition % (0-100)", min_value=0, max_value=100, value=60)
    tyre_name_input = st.text_input("Tyre & OST Name (Optional)")
    process_button = st.button("Process Image", type="primary")

# --- MAIN PAGE LAYOUT ---

# 1. Top Container (Results)
results_spot = st.container()

# 2. Bottom Container (Instructions)
instructions_spot = st.container()

# Logic for Processing
if process_button:
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        with st.spinner('Analyzing...'):
            st.session_state.final_figure = process_image_data(image_bytes, contact_width, threshold, tyre_name_input)
    else:
        st.warning("Please upload an image first.")

# Populate Results (Appears at the top of the page code)
if st.session_state.final_figure is not None:
    with results_spot:
        st.subheader("Analysis Results")
        st.pyplot(st.session_state.final_figure)
        
        plot_buffer = save_plot_to_buffer(st.session_state.final_figure)
        file_name = f"{tyre_name_input if tyre_name_input else 'Tyre_Analysis'}.png"
        
        st.download_button(
            label="💾 Download Result Image",
            data=plot_buffer,
            file_name=file_name,
            mime="image/png"
        )
        st.divider()

# Populate Instructions (Appears at the bottom if result exists, else top)
with instructions_spot:
    st.markdown("## Tyre Contact Area Generator")
    st.markdown(":red[**IMPORTANT INSTRUCTIONS - PLEASE READ BEFORE UPLOADING**]")  

    col_text, col_img1, col_img2 = st.columns([6, 1, 1])
    with col_text:
        st.markdown(""" 
        1. **Image Prep:** Make sure the image is aligned properly, with the maximum contact length in the middle of the horizontal page. Remove all additional or unnecessary black spots.
        2. **Parameters:** Accuracy in contact width is paramount for area calculations. Exercise extreme precision during your measurements.
        3. **Ink Quality:** If you think a contact should be there but the ink is faint, **fill it using Paint/Snipping Tool** before uploading.
        4. **Black Recognition:** Modulate the threshold settings until the processed image achieves optimal clarity and definition.
        """)
        
    with col_img1:
        st.image("image_02.png", caption="✅ Correct Alignment", use_container_width=True)
    with col_img2:
        st.image("image_01.png", caption="❌ Incorrect Alignment", use_container_width=True)
