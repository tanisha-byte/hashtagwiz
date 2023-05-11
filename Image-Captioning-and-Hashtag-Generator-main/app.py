# Libraries
import streamlit as st
from transformers import AutoProcessor
from transformers import AutoTokenizer
from PIL import Image
import requests
from transformers import BlipForConditionalGeneration
import openai
from itertools import cycle
from tqdm import tqdm
import torch
import os
from dotenv import load_dotenv

# Object creation model, tokenizer and processor from HuggingFace
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

load_dotenv()
# Getting the key from env
# openai.api_key = os.environ.get('API_KEY')  ## you Openai key
openai.api_key = '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'
openai_model = "text-davinci-002"  # OpenAI model


def caption_generator(des):
    caption_prompt = ('''Please generate five unique and creative captions to use on Instagram for a photo that shows 
    ''' + des + '''. The captions should be fun and creative and should increase social media reach.
    Captions:
    1.
    2.
    3.
    4.
    5.
    ''')

    # Caption generation
    response = openai.Completion.create(
        engine=openai_model,
        prompt=caption_prompt,
        max_tokens=(175 * 3),
        n=1,
        stop=None,
        temperature=0.7,
    )

    caption = response.choices[0].text.strip().split("\n")
    return (caption)

def prediction(img_list):
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    img = []

    for image in tqdm(img_list):

        i_image = Image.open(image)  # Storing of Image
        st.image(i_image, width=200)  # Display of Image

        if i_image.mode != "RGB":  # Check if the image is in RGB mode
            i_image = i_image.convert(mode="RGB")

        img.append(i_image)  # Add image to the list

    # Image data to pixel values
    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    # Using model to generate output from the pixel values of Image
    output = model.generate(pixel_val, **gen_kwargs)

    # To convert output to text
    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict


def sample():
    # Sample Images in the
    sp_images = {'Sample 1': 'image\\camel.png', 'Sample 2': 'image\\socialmedia.png', 'Sample 3': 'image\\footballer.png'}

    colms = cycle(st.columns(3))  # No of Columns

    for sp in sp_images.values():  # To display the sample images
        next(colms).image(sp, width=150)

    for i, sp in enumerate(sp_images.values()):  # loop to generate caption and hashtags for the sample images

        if next(colms).button("Generate", key=i):  # Prediction is called only on the selected image

            description = prediction([sp])
            st.subheader("Description for the Image:")
            st.write(description[0])

            st.subheader("Captions for this image are:")
            captions = caption_generator(description[0])  # Function to generate caption
            for caption in captions:  # Display Captions
                st.write(caption)

def upload():
    # Form uploader inside tab
    with st.form("uploader"):
        # Image input
        image = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        # Generate button
        submit = st.form_submit_button("Generate")

        if submit:  # submit condition
            description = prediction(image)

            st.subheader("Description of the Image:")
            for i, caption in enumerate(description):
                st.write(caption)

            st.subheader("Captions for this image are:")
            captions = caption_generator(description[0])  # Function call to generate caption
            for caption in captions:  # Present Captions
                st.write(caption)

            # st.subheader("#Hashtags")
            # hashtags = hashtag_generator(description[0])  # Function call to generate hashtag
            # for hash in hashtags:  # Present Hashtags
            #     st.write(hash)


def main():
    # title on the tab
    st.set_page_config(page_title="Caption generation")
    # Title of the page
    st.title("Caption Wizard")

    # Tabs on the page
    tab1, tab2 = st.tabs(["Upload Image", "Sample"])

    # Selection of Tabs
    with tab1:  # Sample images tab
        upload()

    with tab2:  # Upload images tab
        sample()

    # Sub-title of the page
    st.subheader('Tanisha Sharma')


if __name__ == '__main__':
    main()