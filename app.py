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

def dft(input):
    image = input.convert("L")
    image = image.resize((128, 128))
    pixels = np.array(image, dtype=np.float32)

    new_pixels = np.zeros_like(pixels, dtype=complex)

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            total = 0.0
            for x in range(pixels.shape[0]):
                for y in range(pixels.shape[1]):
                    power = -2j * np.pi * ((u*x)/pixels.shape[0] + (v*y)/pixels.shape[1])
                    total += pixels[x,y] * np.exp(power)
            new_pixels[u,v] = total

    dft_pixels = 20 * np.log(np.abs(new_pixels) + 1)
    dft_pixels = dft_pixels - dft_pixels.min()
    dft_pixels = dft_pixels / dft_pixels.max() * 255
    
    dft_img = Image.fromarray(dft_pixels.astype(np.uint8))
    dft_img.show()

    return dft_img

def inverse_dft(input):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    inverse_pixels = np.fft.ifft2(dft_pixels)
    inverse_pixels = np.real(inverse_pixels)

    fft_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    fft_img.show()

    return fft_img

def ideal_low_pass_filter(input, D0):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    new_pixels = np.fft.fftshift(dft_pixels)

    filter = np.zeros_like(pixels, dtype=np.float32)
    M2 = pixels.shape[0] // 2
    N2 = pixels.shape[1] // 2

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            D = np.sqrt((u-M2)**2 + (v-N2)**2)
            if D <= D0:
                filter[u,v] = 1

    low_pixels = new_pixels * filter

    low_pixels = np.fft.ifftshift(low_pixels)
    inverse_pixels = np.fft.ifft2(low_pixels)
    inverse_pixels = np.real(inverse_pixels)

    low_pass_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    low_pass_img.show()

    return low_pass_img

def butterworth_low_pass_filter(input, order, D0):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    new_pixels = np.fft.fftshift(dft_pixels)

    filter = np.zeros_like(pixels, dtype=np.float32)
    M2 = pixels.shape[0] // 2
    N2 = pixels.shape[1] // 2

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            D = np.sqrt((u-M2)**2 + (v-N2)**2)
            transfer = 1 / (1 + np.pow(D/D0, 2*order))
            filter[u,v] = transfer

    low_pixels = new_pixels * filter

    low_pixels = np.fft.ifftshift(low_pixels)
    inverse_pixels = np.fft.ifft2(low_pixels)
    inverse_pixels = np.real(inverse_pixels)

    butterworth_low_pass_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    butterworth_low_pass_img.show()

    return butterworth_low_pass_img

def gaussian_low_pass_filter(input, D0):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    new_pixels = np.fft.fftshift(dft_pixels)

    filter = np.zeros_like(pixels, dtype=np.float32)
    M2 = pixels.shape[0] // 2
    N2 = pixels.shape[1] // 2

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            D = np.sqrt((u-M2)**2 + (v-N2)**2)
            power = -1 * (D**2) / (2*(D0**2))
            filter[u,v] = np.exp(power)

    low_pixels = new_pixels * filter

    low_pixels = np.fft.ifftshift(low_pixels)
    inverse_pixels = np.fft.ifft2(low_pixels)
    inverse_pixels = np.real(inverse_pixels)

    gaussian_low_pass_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    gaussian_low_pass_img.show()

    return gaussian_low_pass_img

def ideal_high_pass_filter(input, D0):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    new_pixels = np.fft.fftshift(dft_pixels)

    filter = np.zeros_like(pixels, dtype=np.float32)
    M2 = pixels.shape[0] // 2
    N2 = pixels.shape[1] // 2

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            D = np.sqrt((u-M2)**2 + (v-N2)**2)
            if D > D0:
                filter[u,v] = 1

    high_pixels = new_pixels * filter

    high_pixels = np.fft.ifftshift(high_pixels)
    inverse_pixels = np.fft.ifft2(high_pixels)
    inverse_pixels = np.real(inverse_pixels)

    high_pass_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    high_pass_img.show()

    return high_pass_img

def butterworth_high_pass_filter(input, order, D0):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    new_pixels = np.fft.fftshift(dft_pixels)

    filter = np.zeros_like(pixels, dtype=np.float32)
    M2 = pixels.shape[0] // 2
    N2 = pixels.shape[1] // 2

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            D = np.sqrt((u-M2)**2 + (v-N2)**2)
            if D:
                transfer = 1 / (1 + np.pow(D0/D, 2*order))
            filter[u,v] = transfer

    high_pixels = new_pixels * filter

    high_pixels = np.fft.ifftshift(high_pixels)
    inverse_pixels = np.fft.ifft2(high_pixels)
    inverse_pixels = np.real(inverse_pixels)

    butterworth_high_pass_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    butterworth_high_pass_img.show()

    return butterworth_high_pass_img

def gaussian_high_pass_filter(input, D0):
    image = input.convert("L")
    pixels = np.array(image)

    dft_pixels = np.fft.fft2(pixels)
    new_pixels = np.fft.fftshift(dft_pixels)

    filter = np.zeros_like(pixels, dtype=np.float32)
    M2 = pixels.shape[0] // 2
    N2 = pixels.shape[1] // 2

    for u in range(pixels.shape[0]):
        for v in range(pixels.shape[1]):
            D = np.sqrt((u-M2)**2 + (v-N2)**2)
            power = -1 * (D**2) / (2*(D0**2))
            filter[u,v] = 1 - np.exp(power)

    high_pixels = new_pixels * filter

    high_pixels = np.fft.ifftshift(high_pixels)
    inverse_pixels = np.fft.ifft2(high_pixels)
    inverse_pixels = np.real(inverse_pixels)

    gaussian_high_pass_img = Image.fromarray(inverse_pixels.astype(np.uint8))
    gaussian_high_pass_img.show()

    return gaussian_high_pass_img

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
    "Sharpening (Laplacian)", "Edge Detection (Sobel)",
    "Discrete Fourier Transform", "Inverse Discrete Fourier Transform",
    "Ideal Low Pass Filter", "Butterworth Low Pass Filter", 
    "Gaussian Low Pass Filter", "Ideal High Pass Filter", 
    "Butterworth High Pass Filter", "Gaussian High Pass Filter"
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
    elif operation == "Ideal Low Pass Filter":
        radius = st.slider("Radius", 0, 300, 50)
    elif operation == "Butterworth Low Pass Filter":
        order = st.select_slider("Order", options=[1, 2, 3, 4], value=2)
        radius = st.slider("Radius", 0, 300, 50)
    elif operation == "Gaussian Low Pass Filter":
        radius = st.slider("Radius", 0, 300, 50)
    elif operation == "Ideal High Pass Filter":
        radius = st.slider("Radius", 0, 300, 50)
    elif operation == "Butterworth High Pass Filter":
        order = st.select_slider("Order", options=[1, 2, 3, 4], value=2)
        radius = st.slider("Radius", 0, 300, 50)
    elif operation == "Gaussian High Pass Filter":
        radius = st.slider("Radius", 0, 300, 50)

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
            elif operation == "Discrete Fourier Transform":
                result = dft(image)
            elif operation == "Inverse Discrete Fourier Transform":
                result = inverse_dft(image)
            elif operation == "Ideal Low Pass Filter":
                result = ideal_low_pass_filter(image, radius)
            elif operation == "Butterworth Low Pass Filter":
                result = butterworth_low_pass_filter(image, order, radius)
            elif operation == "Gaussian Low Pass Filter":
                result = gaussian_low_pass_filter(image, radius)
            elif operation == "Ideal High Pass Filter":
                result = ideal_high_pass_filter(image, radius)
            elif operation == "Butterworth High Pass Filter":
                result = butterworth_high_pass_filter(image, order, radius)
            elif operation == "Gaussian High Pass Filter":
                result = gaussian_high_pass_filter(image, radius)
    
    with col2:
        if result is not None:
            st.image(result, caption="Processed Image", use_container_width=True)
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button("Download Result", buf.getvalue(), file_name="result.png")
        else:
            st.write("Processed image will appear here after you click Apply.")