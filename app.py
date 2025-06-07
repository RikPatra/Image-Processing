import numpy as np
import matplotlib.pyplot as plt
import math
import os
import io
import sys
import streamlit as st
from PIL import Image

def upscale(input, scale):
    image = input.convert("RGB")
    width, height = image.size
    pixels = image.load()

    up_width = int(width*scale)
    up_height = int(height*scale)

    upscaled = Image.new(mode="RGB", size=(up_width, up_height))
    up_pixels = upscaled.load()

    for y in range(up_height):
        for x in range(up_width):
            old_x = int(x/scale)
            old_y = int(y/scale)
            up_pixels[x,y] = pixels[old_x,old_y]

    upscaled.show()

    return upscaled

def downscale(input, scale):
    image = input.convert("RGB")
    width, height = image.size
    pixels = image.load()

    down_width = int(width/scale)
    down_height = int(height/scale)

    downscaled = Image.new(mode="RGB", size=(down_width, down_height))
    down_pixels = downscaled.load()

    for y in range(down_height):
        for x in range(down_width):
            old_x = int(x*scale)
            old_y = int(y*scale)
            down_pixels[x,y] = pixels[old_x,old_y]

    downscaled.show()

    return downscaled

def quantization(input, levels):
    image = input.convert("L")
    image_array = np.array(image)

    factor = 256//levels
    quant_array = (image_array//factor)*factor

    quantized = Image.fromarray(quant_array.astype(np.uint8))
    quantized.show()

    return quantized

def histogram_equalisation(input):
    image = input.convert("L")
    image.show()
    hist = image.histogram()
    
    cdf = []
    cumsum = 0
    for i in hist:
        cumsum += i
        cdf.append(cumsum)
    
    cdf = [i/cumsum for i in cdf]
    
    cdf = [round(255*i) for i in cdf]
    
    pixels = np.array(image)
    new_pixels = np.zeros_like(pixels)
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            new_pixels[x,y] = cdf[pixels[x,y]]

    equalized = Image.fromarray(new_pixels)
    equalized.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity (0–255)")
    plt.ylabel("Frequency")
    plt.plot(hist, color='black')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Intensity (0–255)")
    plt.ylabel("Frequency")
    plt.plot(equalized.histogram(), color='blue')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return equalized

def negative(input):
    image = input.convert("L")
    pixels = np.array(image)

    pixels = pixels / 255.0

    neg_pixels = 255 * (1 - pixels)

    negative_img = Image.fromarray(neg_pixels.astype(np.uint8))
    negative_img.show()

    return negative_img

def thresholding(input, threshold):
    image = input.convert("L")
    pixels = np.array(image)

    pixels = pixels / 255.0
    
    new_pixels = np.zeros_like(pixels)
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            if pixels[x,y] > threshold:
                new_pixels[x,y] = 255

    thresholded = Image.fromarray(new_pixels.astype(np.uint8))
    thresholded.show()

    return thresholded

def logarithmic(input, c):
    image = input.convert("L")
    pixels = np.array(image)

    pixels = pixels / 255.0
    
    new_pixels = np.zeros_like(pixels)
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            new_pixels[x,y] = c * math.log(1 + pixels[x,y])

    log_img = Image.fromarray(new_pixels.astype(np.uint8))
    log_img.show()

    return log_img

def power_law(input, gamma):
    image = input.convert("L")
    pixels = np.array(image)

    pixels = pixels / 255.0
    
    new_pixels = np.zeros_like(pixels)
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            new_pixels[x,y] = 255 * math.pow(pixels[x,y], gamma)

    pow_img = Image.fromarray(new_pixels.astype(np.uint8))
    pow_img.show()

    return pow_img

def bit_plane_slicing(input, bit):
    image = input.convert("L")
    pixels = np.array(image)

    bit_pixels = (pixels >> bit) & 1
    bit_pixels = bit_pixels * 255

    bit_img = Image.fromarray(bit_pixels.astype(np.uint8))
    bit_img.show()

    return bit_img

st.title("Image Processing Application")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    operation = st.selectbox("Choose operation", [
        "Upscale", "Downscale", "Quantization",
        "Histogram Equalization", "Negative",
        "Thresholding", "Logarithmic", "Power Law",
        "Bit Plane Slicing"
    ])

    if operation == "Upscale":
        scale = st.slider("Scale", 1.0, 5.0, 2.0)
        result = upscale(image, scale)
    elif operation == "Downscale":
        scale = st.slider("Scale", 0.1, 1.0, 0.5)
        result = downscale(image)
    elif operation == "Quantization":
        bits = st.select_slider("Bits", options=[1, 2, 3, 4, 5, 6, 7, 8], value=4)
        levels = math.pow(2, bits)
        result = quantization(image, levels)
    elif operation == "Histogram Equalization":
        result = histogram_equalisation(image)
    elif operation == "Negative":
        result = negative(image)
    elif operation == "Thresholding":
        threshold = st.slider("Threshold", 0.1, 1.0, 0.5)
        result = thresholding(image, threshold)
    elif operation == "Logarithmic":
        c = st.slider("c-Value", 1, 255, 255)
        result = logarithmic(image, c)
    elif operation == "Power Law":
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0)
        result = power_law(image, gamma)
    elif operation == "Bit Plane Slicing":
        bit = st.select_slider("Bit", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3)
        result = bit_plane_slicing(image, bit)

    if uploaded_file and st.button("Apply"):
        st.image(result, caption="Processed Image", use_container_width=True)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), file_name="result.png")