import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO


def single_image_run(HSI, MSI):
    IMAGE_SIZE_MSI = 64
    IMAGE_SIZE_HSI = 32

    NUM_BANDS = 40
    HSI_bands = []
    HSI_arrays = []
    MSI_bands = []
    MSI_arrays = []

    for i in range(0, 40):
        HSI_bands.append(HSI[i])

    HSI_array = np.dstack(HSI_bands)
    HSI_arrays.append(HSI_array)

    for i in range(MSI.shape[0]):
        MSI_bands.append(MSI[i])

    MSI_array = np.dstack(MSI_bands)
    MSI_arrays.append(MSI_array)

    MSI_arrays = np.array(MSI_arrays)
    HSI_arrays = np.array(HSI_arrays)

    model = load_model("F:\FYP\HSRnet-coreimages\models\HSR_Net9_final_100e.h5")

    pred = model.predict(
        [
            HSI_arrays.reshape(1, IMAGE_SIZE_HSI, IMAGE_SIZE_HSI, NUM_BANDS),
            MSI_arrays.reshape(1, IMAGE_SIZE_MSI, IMAGE_SIZE_MSI, 3),
        ]
    )

    pred = np.transpose(pred[0], (2, 0, 1))

    return pred


# Login credentials
credentials = pd.DataFrame(
    {"Username": ["user1", "user2"], "Password": ["pass1", "pass2"]}
)

# Registration records
registrations = pd.DataFrame(columns=["Username", "Password"])


# Login page
placeholder = st.empty()


def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if (
        credentials.query("Username == @username and Password == @password").shape[0]
        > 0
    ):
        placeholder.empty()

        st.success("Login Successful")
        st.balloons()

        landing_page()
    else:
        st.error("Invalid credentials")


# Registration page
def register():
    st.title("Register")

    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")

    # Add to registration record
    if st.button("Register"):
        new_registration = pd.DataFrame(
            {"Username": [new_username], "Password": [new_password]}
        )
        registrations = registrations.append(new_registration, ignore_index=True)
        st.success("Registered successfully")


# Landing page
def landing_page():
    st.title("Welcome")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")

    st.write("This app allows performing operations like:")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("- HSI Super Resolution")
        st.write(
            "Hyperspectral Image (HSI) Super Resolution refers to the process of fusing a high spectral, low spatial resolution HSI with a high spatial, low spectral resolution multispectral image (MSI) to generate a super resolved image that has both high spectral and high spatial resolutions."
        )

    with c2:
        st.write("- HSI Denoising")
        st.write(
            "Hyperspectral Image (HSI) Denoising refers to the process of removing noise from HSI data while preserving the original spectral and spatial details as much as possible."
        )

    with c3:
        st.write("- HSI Co-Registration")
        st.write(
            "Hyperspectral Image (HSI) Co-Registration refers to the alignment of HSIs with MSIs capturing the same scene taken at different times, from different viewpoints, or by different sensors."
        )

    st.write("\n")
    st.write("\n")
    st.write("\n")
    choice = st.radio(
        "Select operation",
        ["HSI Super Resolution", "HSI Denoising", "HSI Co-Registration"],
    )

    if choice == "HSI Super Resolution":
        super_resolution()
    elif choice == "HSI Denoising":
        denoising()
    elif choice == "HSI Co-Registration":
        co_registration()


# Super resolution page


def super_resolution():
    @st.cache_resource
    def load_npy(uploaded_file):
        bytes_data = uploaded_file.getvalue()
        data = np.load(BytesIO(bytes_data))
        return data

    LRHSI = st.file_uploader("Low Resolution HSI", type=["npy"])
    if LRHSI:
        LRHSI = load_npy(LRHSI)

    HRMSI = st.file_uploader("High Resolution MSI", type=["npy"])
    if HRMSI:
        HRMSI = load_npy(HRMSI)

    LRHSI = np.array(LRHSI)
    HRMSI = np.array(HRMSI)

    if LRHSI.any() and HRMSI.any():
        if st.button("Start Operation"):
            output = single_image_run(LRHSI, HRMSI)

            # Create columns
            #col1, col2, col3, col4, col5 = st.columns(5)
            col1, col2, col3= st.columns(3)

            with col1:
                st.write("Low Resolution Image")
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(LRHSI[0])
                ax.axis("off")
                st.pyplot(fig)

            with col3:
                st.write("Super Resolved Image")
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(output[0])
                ax.axis("off")
                st.pyplot(fig)

                if st.button("Save Output"):
                    np.save("output.npy", output)
                    st.success("Output image saved")


# Other pages
def denoising():
    pass


def co_registration():
    pass


# Main app
st.sidebar.title("Menu")
choice = st.sidebar.radio("", ["Login", "Register", "Land"])
if choice == "Login":
    login()
elif choice == "Register":
    register()
elif choice == "Land":
    landing_page()
