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

def smoothening(input):
    image = input.convert("L")
    pixels = np.array(image)

    filter = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ], dtype=np.float32) / 9.0
    
    pad_pixels = np.pad(pixels, ((1,1),(1,1)), mode="edge")

    new_pixels = np.zeros_like(pixels, dtype=np.float32)

    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            region = pad_pixels[x:x+3, y:y+3]
            avg = np.sum(region * filter)
            new_pixels[x,y] = avg

    new_pixels = np.clip(new_pixels, 0, 255)

    smoothed = Image.fromarray(new_pixels.astype(np.uint8))
    smoothed.show()

    return smoothed

def laplacian(input):
    image = input.convert("L")
    pixels = np.array(image)

    filter = np.array([
        [ 0,-1, 0],
        [-1, 5,-1],
        [ 0,-1, 0]
    ], dtype=np.float32)
    
    pad_pixels = np.pad(pixels, ((1,1),(1,1)), mode="edge")

    new_pixels = np.zeros_like(pixels, dtype=np.float32)

    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            region = pad_pixels[x:x+3, y:y+3]
            avg = np.sum(region * filter)
            new_pixels[x,y] = avg

    new_pixels = np.clip(new_pixels, 0, 255)

    sharpened = Image.fromarray(new_pixels.astype(np.uint8))
    sharpened.show()

    return sharpened

def sobel(input):
    image = input.convert("L")
    pixels = np.array(image)

    filter1 = np.array([
        [-1,-2,-1],
        [ 0, 0, 0],
        [ 1, 2, 1]
    ], dtype=np.float32)

    filter2 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    pad_pixels = np.pad(pixels, ((1,1),(1,1)), mode="edge")

    new_pixels = np.zeros_like(pixels, dtype=np.float32)

    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            region = pad_pixels[x:x+3, y:y+3]
            avg1 = np.sum(region * filter1)
            avg2 = np.sum(region * filter2)
            avg = np.sqrt(avg1**2 + avg2**2)
            new_pixels[x,y] = avg

    new_pixels = np.clip(new_pixels, 0, 255)

    sobeled = Image.fromarray(new_pixels.astype(np.uint8))
    sobeled.show()

    return sobeled

st.set_page_config(layout="wide")
st.title("emagine - Image Processing Tool")
st.header("IPCV Internal Assessment by Rik Patra")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

st.sidebar.title("Operations")
operation = st.sidebar.radio("Choose operation", [
    "Upscale", "Downscale", "Quantization",
    "Histogram Equalization", "Negative",
    "Thresholding", "Logarithmic", "Power Law",
    "Bit Plane Slicing", "Smoothening (Simple Averaging)",
    "Sharpening (Laplacian)", "Edge Detection (Sobel)"
])

if uploaded_file:
    image = Image.open(uploaded_file)

    if operation == "Upscale":
        scale = st.slider("Scale", 1.0, 5.0, 2.0)
    elif operation == "Downscale":
        scale = st.slider("Scale", 0.1, 1.0, 0.5)
    elif operation == "Quantization":
        bits = st.select_slider("Bits", options=[1, 2, 3, 4, 5, 6, 7, 8], value=4)
        levels = math.pow(2, bits)
    elif operation == "Thresholding":
        threshold = st.slider("Threshold", 0.1, 1.0, 0.5)
    elif operation == "Logarithmic":
        c = st.slider("c-Value", 1, 255, 255)
    elif operation == "Power Law":
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0)
    elif operation == "Bit Plane Slicing":
        bit = st.select_slider("Bit", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        apply = st.button("Apply")
        result = None
        if apply:
            if operation == "Upscale":
                result = upscale(image, scale)
            elif operation == "Downscale":
                result = downscale(image)
            elif operation == "Quantization":
                result = quantization(image, levels)
            elif operation == "Histogram Equalization":
                result = histogram_equalisation(image)
            elif operation == "Negative":
                result = negative(image)
            elif operation == "Thresholding":
                result = thresholding(image, threshold)
            elif operation == "Logarithmic":
                result = logarithmic(image, c)
            elif operation == "Power Law":
                result = power_law(image, gamma)
            elif operation == "Bit Plane Slicing":
                result = bit_plane_slicing(image, bit)
            elif operation == "Smoothening (Simple Averaging)":
                result = smoothening(image)
            elif operation == "Sharpening (Laplacian)":
                result = laplacian(image)
            elif operation == "Edge Detection (Sobel)":
                result = sobel(image)
    
    with col2:
        if result is not None:
            st.image(result, caption="Processed Image", use_container_width=True)
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button("Download Result", buf.getvalue(), file_name="result.png")
        else:
            st.write("Processed image will appear here after you click Apply.")