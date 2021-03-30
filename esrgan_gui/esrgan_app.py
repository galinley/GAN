import sys
import streamlit as st
from PIL import Image
import numpy as np
import time
import base64
import esrgan


def main():
    # Setup the main UI
    #st.image('background.png')
    st.title("Enhanced Super Resolution GAN Demo")

    # Load the ESRGAN model
    model = load_esrgan_model()

    selected_box = st.sidebar.selectbox(
        'Choose one of the following:',
        ('Welcome', 'Upscale An Image', 'How it Works', 'FAQ')
    )
    if selected_box == 'Welcome':
        welcome()
    if selected_box == 'Upscale An Image':
        upscale(model)
    if selected_box == 'How it Works':
        how_it_works()
    if selected_box == 'FAQ':
        faq()


# Ensure that load_esrgan_model is called only once, when the app first loads.
@st.cache(allow_output_mutation=True)
def load_esrgan_model():
    return esrgan.load_model()


def welcome():
    st.markdown(">This app upscales a low resolution image to a high "
                "resolution image \n"
                "using a TensorFlow implementation of ESRGAN.\n\n")


def upscale(model):
    st.markdown(">Select the low resolution image you'd like to upscale.")
    uploaded_lr_img = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

    if uploaded_lr_img is not None:
        lr_img = Image.open(uploaded_lr_img)
        st.image(lr_img)
        left_column, center_column, right_column = st.beta_columns(3)
        st.write("Uploaded image resolution: ", lr_img.size)

        # TODO: Add model interpolation
        if st.sidebar.checkbox('Show advanced options'):
                interp = st.sidebar.slider("Model Interpolation", 0, 100, 50, 5)

        upscale_methods = st.sidebar.checkbox('Compare Upscale Methods (optional)')
        if upscale_methods:
            cb_nearest = st.sidebar.checkbox('Nearest Neighbor')
            cb_bicubic = st.sidebar.checkbox('Bicubic')

        if st.button('Upscale Image'):
            with st.spinner("Upscaling image..."):
                hr_image, processing_time = esrgan.upscale_image(lr_img, model)
                hr_size = hr_image.size
                st.write("Processing time: ", processing_time)
                st.write("New upscaled resolution: ", hr_size)

                st.markdown(">***ESRGAN* Super Resolution image.**")
                st.image(hr_image)

            if upscale_methods:
                if cb_nearest:
                    nearest = lr_img.resize(size=(lr_img.size[0] * 4, lr_img.size[1] * 4), resample=Image.NEAREST)
                    st.markdown(">**Resized image using *Nearest Neighbor* resample method**.")
                    st.image(nearest)
                if cb_bicubic:
                    bicubic = lr_img.resize(size=(lr_img.size[0] * 4, lr_img.size[1] * 4), resample=Image.BICUBIC)
                    st.markdown(">**Resized image using *Bicubic* resample method.**")
                    st.image(bicubic)


def how_it_works():
    # TODO: Add section on how ESRGAN works
    st.markdown(">Coming soon...\n\n")
    left_column, center_column, right_column = st.beta_columns(3)
    pressed = center_column.button('Press me?')
    if pressed:
        right_column.write("Woohoo!")

def faq():
    st.markdown(">Coming soon...\n\n")


def display_bg_image():
    main_bg = "background.png"
    main_bg_ext = "png"

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()