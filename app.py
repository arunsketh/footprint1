import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from copy import deepcopy
import matplotlib.pyplot as plt
import io

# --- Core Logic Functions ---

def painter(array):
    """Fills vertical gaps (white pixels) that are 'trapped' under black pixels."""
    arr = deepcopy(array)
    rows, cols = arr.shape
    for j in range(cols):
        found_black = False
        for i in range(rows):
            if not arr[i][j]:  # False is Black in bool array
                found_black = True
            elif arr[i][j] and found_black:
                arr[i][j] = False  # Fill white gap
    return arr

def process_image_data(image_bytes, contact_width_mm, threshold_percent, tyre_name):
    try:
        # 1. Load and Pre-process
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        blur = img.filter(ImageFilter.MedianFilter(size=7))
        im_array = np.array(blur)

        # 2. Dynamic Thresholding
        threshold_limit = int(float(threshold_percent) * 2.55)
        # Binary mask: True for white/background, False for black/tyre
        binary_mask = np.where(im_array <= threshold_limit, False, True)

        # 3. Find Footprint Bounding Box
        black_coords = np.argwhere(binary_mask == False)
        if black_coords.size == 0:
            st.error("No footprint detected. Please lower the Black Recognition %.")
            return None

        y_min, x_min = black_coords.min(axis=0)
        y_max, x_max = black_coords.max(axis=0)

        # 4. Define Midpoints based on footprint dimensions
        mid_y = (y_min + y_max) // 2
        mid_x = (x_min + x_max) // 2

        # 5. Extract Quadrants
        # Top-Left, Top-Right, Bottom-Left, Bottom-Right
        q_tl = binary_mask[y_min:mid_y, x_min:mid_x]
        q_tr = binary_mask[y_min:mid_y, mid_x:x_max]
        q_bl = binary_mask[mid_y:y_max, x_min:mid_x]
        q_br = binary_mask[mid_y:y_max, mid_x:x_max]

        # 6. Fill Gaps (Center-Outward Logic)
        # Top quadrants: flip vertically, paint, then flip back
        filled_tl = np.flipud(painter(np.flipud(q_tl)))
        filled_tr = np.flipud(painter(np.flipud(q_tr)))
        
        # Bottom quadrants: paint normally
        filled_bl = painter(q_bl)
        filled_br = painter(q_br)

        # 7. Reconstruct Filled Footprint
        top_half = np.hstack((filled_tl, filled_tr))
        bottom_half = np.hstack((filled_bl, filled_br))
        filled_footprint = np.vstack((top_half, bottom_half))

        # 8. Calculations
        # Scaling based on user-provided width
        footprint_pixel_width = x_max - x_min
        footprint_pixel_height = y_max - y_min
        
        px_per_mm = footprint_pixel_width / contact_width_mm
        contact_length_mm = round(footprint_pixel_height / px_per_mm, 1)

        # Area Calculations (1 mm^2 = 0.00155 sq.in)
        net_pixels = np.count_nonzero(binary_mask[y_min:y_max, x_min:x_max] == False)
        gross_pixels = np.count_nonzero(filled_footprint == False)

        net_area_in2 = round((net_pixels / (px_per_mm**2)) * 0.00155, 2)
        gross_area_in2 = round((gross_pixels / (px_per_mm**2)) * 0.00155, 2)

        # 9. Visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        
        # Original with Bounding Box
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("1. Original & Detected Bounds", color='blue', fontweight='bold')
        ax[0].add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, color='red', fill=False, lw=2))
        ax[0].axis('off')

        # Processed (Net Area)
        ax[1].imshow(binary_mask[y_min:y_max, x_min:x_max], cmap='gray')
        ax[1].set_title("2. Net Contact Area", color='green', fontweight='bold')
        ax[1].text(0.5, -0.1, f"Net: {net_area_in2} sq.in", transform=ax[1].transAxes, ha='center', fontsize=12)
        ax[1].axis('off')

        # Filled (Gross Area)
        ax[2].imshow(filled_footprint, cmap='gray')
        ax[2].set_title("3. Gross (Filled) Area", color='red', fontweight='bold')
        ax[2].text(0.5, -0.1, f"Gross: {gross_area_in2} sq.in\nLength: {contact_length_mm} mm", 
                   transform=ax[2].transAxes, ha='center', fontsize=12)
        ax[2].axis('off')

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- Streamlit Interface ---

st.set_page_config(layout="wide", page_title="Tyre Footprint Analyzer")
st.title("🛞 Tyre Contact Area Analyzer")
st.markdown("---")

if 'result_fig' not in st.session_state:
    st.session_state.result_fig = None

with st.sidebar:
    st.header("Upload & Parameters")
    file = st.file_uploader("Upload Tyre Print", type=['jpg', 'jpeg', 'png'])
    c_width = st.number_input("Known Contact Width (mm)", value=150.0, step=1.0)
    thresh = st.slider("Black Recognition Sensitivity", 0, 100, 60)
    name = st.text_input("Tyre ID / Label")
    
    if st.button("Run Analysis", type="primary"):
        if file:
            with st.spinner("Processing..."):
                st.session_state.result_fig = process_image_data(file.getvalue(), c_width, thresh, name)
        else:
            st.warning("Please upload an image.")

if st.session_state.result_fig:
    if name:
        st.subheader(f"Results for: {name}")
    st.pyplot(st.session_state.result_fig)
