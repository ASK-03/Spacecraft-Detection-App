import streamlit as st
import cv2
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import numpy as np

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("./model/best.pt", task="detect")  # Model without quantization
model_lite = YOLO("./model/best.torchscript", task="detect")  # Model with quantization


def detect_spacecraft(image):
    results = model.predict(image)
    results_lite = model_lite.predict(image)

    res_image = None
    res_lite_image = None
    for r in results:
        res_image = r.plot()
        break

    for r in results_lite:
        res_lite_image = r.plot()
        break

    temp_res_speed = results[0].speed
    temp_res_lit_speed = results_lite[0].speed

    total_res_speed = (
        temp_res_speed["preprocess"]
        + temp_res_speed["inference"]
        + temp_res_speed["postprocess"]
    )
    total_res_lit_speed = (
        temp_res_lit_speed["preprocess"]
        + temp_res_lit_speed["inference"]
        + temp_res_lit_speed["postprocess"]
    )

    return [
        {"image": res_image, "speed": total_res_speed},
        {"image": res_lite_image, "speed": total_res_lit_speed},
    ]


def main():
    """
    Space Craft Detection App
    """
    st.title("Space Craft Detection App")
    st.text("Build with Streamlit,YoloV8 and OpenCV")

    menu = ["About", "Metrics", "Detection"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Detection":
        st.subheader("Space Craft Detection")
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        our_image = None
        new_img = None

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

        enhance_type = st.sidebar.radio(
            "Enhance Type",
            ["Original", "Contrast", "Brightness", "Blurring"],
        )

        if our_image is None:
            st.warning("Upload an image")
        else:
            # if enhance_type == "Gray-Scale":
            #     new_img = np.array(our_image.convert("RGB"))
            #     new_img = cv2.cvtColor(new_img, 1)
            #     new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            #     # st.write(new_img)
            #     st.image(new_img)

            if enhance_type == "Contrast":
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(our_image)
                new_img = enhancer.enhance(c_rate)
                st.image(new_img)

            if enhance_type == "Brightness":
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(our_image)
                new_img = enhancer.enhance(c_rate)
                st.image(new_img)

            if enhance_type == "Blurring":
                new_img = np.array(our_image.convert("RGB"))
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
                img = cv2.cvtColor(new_img, 1)
                new_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
                st.image(new_img)
            else:
                pass

            # Space Craft Detection
            task = ["Space Craft"]
            feature_choice = st.sidebar.selectbox("Find Features", task)
            if st.button("Process"):
                if feature_choice == "Space Craft":
                    if new_img is None:
                        new_img = np.array(our_image.convert("RGB"))
                    result_img, result_lite_img = detect_spacecraft(new_img)
                    if result_img is None or result_lite_img is None:
                        exit()
                    st.image(
                        result_img["image"],
                        caption=f'YoloV8 without qunatization (speed: {round(result_img["speed"], 2)})',
                    )
                    st.image(
                        result_lite_img["image"],
                        caption=f'YoloV8 with quantization (speed: {round(result_lite_img["speed"], 2)})',
                    )

                else:
                    st.markdown("## Space Craft not Detected")
                    st.markdown("## Kindly upload correct image")

    elif choice == "Metrics":
        st.subheader("Metrics")
        data = {
            "Metric": [
                "mAP (Mean Average Precision)",
                "Average Loading Time",
                "Average Inference Time",
            ],
            "YoloV8 (no quantization)": [0.52, "3.5 seconds", "~ 50 ms"],
            "YoloV8 (quantization)": [0.51, "1 seconds", "~ 20 ms"],
        }

        st.dataframe(data)

    elif choice == "About":
        st.markdown("## **About**")
        st.write(
            """Spacecraft inspection is the process of closely examining a spacecraft in orbit to assess its condition and functionality. Current spacecraft inspection methods often involve time-consuming and hazardous astronaut-led assessments or the use of expensive LIDAR sensors and robotic arms.  
The project aims to develop algorithms for use on inspector spacecraft to take and process photos of other ships in space. The solution will focus on identifying the boundaries of generic spacecraft in photos. The key operational challenges include handling diverse and potentially damaged spacecraft types in the dataset and developing solutions that run efficiently on a simulated NASA R5 spacecraft's small computer board within a specific code execution platform.
"""
        )

        st.write(
            "[**Project Proposal Link**](https://docs.google.com/document/d/1h9pxsEOmdK9RTldzUgXKXlmzvPBVfFK5gxJ7H5QNkiI/edit)"
        )

        # Add information about the creators
        st.markdown("## **Creators:**")
        st.write("- [Abhishek Singh Kushwaha](https://github.com/ASK-03)")
        st.write("- [Kriti Gupta](https://github.com/Kriti1106)")  # Add more creators if needed


if __name__ == "__main__":
    main()
