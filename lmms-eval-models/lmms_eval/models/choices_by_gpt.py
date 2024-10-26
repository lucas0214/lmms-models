import json
import sys

sys.path.append('/root/new_codes/lmms-eval/lmms_eval/models')

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests


def check_characters(s):
    characters = ['A', 'B', 'C', 'D', 'E']
    results = {char: (char in s) for char in characters}
    return results

"""
def choice_models(base64Frames, caption, options):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "You are a video expert, and I will provide you with a set of video frames along with their corresponding descriptions, as well as several video processing method options. The processing method options are presented to you in the form of a Python dictionary, where the keys are A, B, C, D, E, and the values are the visual processing methods. Your task is to select the most suitable processing methods based on these video frames and descriptions, choosing at least two. Please rank the selected methods from most to least important. You must strictly select from the given options—do not propose any methods outside of the provided choices. Here are the video descriptions ({}) and the visual processing options ({}). Please respond using only the keys (A, B, C, D, or E) of the selected methods. Do not provide explanations for your choices or any additional information.".format(caption, options),
                *map(lambda x: {"image": x, "resize": 768}, base64Frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }

    result = client.chat.completions.create(**params)

    return result.choices[0].message.content
"""

def result_caption(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)

    for da_caption in data:
        if da_caption["video_path"] == video_path:
            caption = da_caption["caption"]
            return caption

    return ""


def result_detect(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)

    for da_caption in data:
        if da_caption["video_path"] == video_path:
            caption = da_caption["caption"]
            return caption

    return None

def result_diff(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)

    for da_caption in data:
        if da_caption["video_path"] == video_path:
            caption = da_caption["reason_diff"]
            return caption
    return ""

def result_ocr(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)

    for da_caption in data:
        if da_caption["video_path"] == video_path:
            caption = da_caption["ocr_result"]
            return caption
    return ""

def result_action(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)

    for da_caption in data:
        if da_caption["video_path"] == video_path:
            caption = da_caption["action_result"]
            return caption
    return ""

def result_audio(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)

    for da_caption in data:
        if da_caption["video_path"] == video_path:
            caption = da_caption["audio_information"]
            return caption
    return ""
def choice_models(json_path, video_path):
    with open(json_path, "r") as caption_file:
        data = json.load(caption_file)
    value = next((d[video_path] for d in data if video_path in d), "Not found")

    return value

def model_main(video_path):

    video_caption = result_caption("../data/captions.json", video_path)

    addition_information = "To help you better understand the information in the video, we have provided additional visual information from the video based on some other models, as follows: The video caption displays " + video_caption + " "

    options_dict = {
        "A": "Detecting objects within video frames",
        "B": "Reasoning and summarizing based on differences between frames",
        "C": "Conducting OCR operations on video frames",
        "D": "Recognizing actions in video content",
        "E": "Synthesizing information via auditory reasoning"
    }

    choice_path = "../data/formatted_choices.json"
    choices = choice_models(choice_path, video_path)  # str

    choice_dict = check_characters(choices)  # dict

    if choice_dict["A"] == "true":
        a_path = "../data/classname.json"
        class_name = result_detect(a_path, video_path)  # list  str

        if class_name is not None:
            cname = ""
            for c in range(len(class_name)):
                if c != (len(class_name)-1):
                    cname = cname + class_name[c] +", "
                else:
                    cname = cname + class_name[c] + ". "

            addition_content = "Based on the object detection method, the video mainly contains objects such as " + cname + "Based on the detected category names from the video frames, semantic expansion is carried out by describing their attributes, uses, and behaviors in different contexts, which are described as follows."

            addition_information = addition_information + addition_content + " "

    if choice_dict["B"] == "true":
        b_path = "../data/reason_b_frame.json"
        diff_d = result_diff(b_path, video_path)
        if diff_d != "":
            addition_reason = "Based on reasoning from the differences between the current frame and the frame four frames later in the video, it can be concluded that " + reason_content

        addition_information = addition_information + addition_reason + " "

    if choice_dict["C"] == "true":
        c_path = "../data/ocr_result.json"
        ocr_o = result_ocr(c_path, video_path)
        if ocr_o != "[]":
            addition_ocr = "Using an OCR model to recognize video frames, the recognition results include "
            l = len(ocr_o) - 1
            addition_ocr = addition_ocr + ocr_o[1:l] + "."
            addition_information = addition_information + addition_ocr + " "

    if choice_dict["D"] == "true":
        d_path = "../data/action_result.json"
        action_c = result_action(d_path, video_path)

        if action_c is not None:
            addition_action = "Using an action recognition model to analyze the video reveals that the main actions include " + action_c
            addition_information = addition_information + addition_action + " "


    if choice_dict["E"] == "true":
        e_path = "../data/audio_information.json"
        audio_a = result_audio(e_path, video_path)

        if audio_a is not None:
            addition_audio = "It can be inferred through deep analysis of the audio information in the video that " + audio_a
            addition_information = addition_information + addition_audio

    return addition_information