import sys
sys.path.append('/root/new_codes/lmms-eval/lmms_eval/models')

from choices_by_gpt import model_main

import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image
import warnings
import cv2

# 忽略所有警告
warnings.filterwarnings("ignore")


API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger

if API_TYPE == "openai":
    # API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


@register_model("gpt4v")
class GPT4V(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        modality: str = "video",
        max_frames_num: int = 32,
        timeout: int = 300,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str


    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames
    """

    def encode_video(self, video_path, for_get_frames_num, save_dir, output_video_path, fps=30):
        output_video_path = save_dir + "/" + output_video_path + ".mp4"
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # 确保最后一帧被包含
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []

        # 确保保存帧的目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 获取帧的高度和宽度（假设所有帧的大小相同）
        height, width, _ = frames[0].shape

        # 初始化视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        frame_lists = []
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)

            # 保存每一帧到指定的目录中，命名为 frame_0.png, frame_1.png...
            frame_save_path = os.path.join(save_dir, f"frame_{i}.png")
            img.save(frame_save_path)
            frame_lists.append(str(frame_save_path))

            # 将图像编码为base64字符串
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

            # 将帧写入视频
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 将RGB转换为BGR以适应cv2
            video_writer.write(frame_bgr)

        # 释放视频写入对象
        video_writer.release()

        return base64_frames, frame_lists
    """

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue
            my_dict = {}
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            additions = []
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    information = model_main(visual)
                    additions.append(information)
                    frames = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)

            my_dict["question"] = contexts
            payload = {"messages": []}
            if API_TYPE == "openai":
                payload["model"] = self.model_version

            response_json = {"role": "user", "content": []}
            # When there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][0]["content"].append({"type": "text", "text": contexts + " " + additions[0]})
                # print(payload)
                for img in imgs:
                    payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx] + " " + additions[idx]})
                    payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the payload
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1] + " " + additions[-1]})
                # print(payload)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            for attempt in range(5):
                try:
                    response = url_requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=self.timeout)
                    response_data = response.json()

                    response_text = response_data["choices"][0]["message"]["content"].strip()
                    my_dict["answer"] = response_text
                    with open("/root/new_codes/answer_gpt4o.json", "a") as json_file:   # 保存输出
                        json.dump(my_dict, json_file)
                        json_file.write(",\n")

                    break  # If successful, break out of the loop

                except Exception as e:
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = ""

                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                    if attempt <= 5:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty string
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {response.json()}")
                        response_text = ""
            res.append(response_text)
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT4V")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
