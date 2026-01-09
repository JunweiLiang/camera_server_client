# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
sys.path.append('third_party/Matcha-TTS')
from cosyvoice_vllm.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice_vllm.utils.file_utils import load_wav

from vllm import ModelRegistry
from cosyvoice_vllm.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


# ----junwei
@app.get("/inference_zero_shot_fast")
@app.post("/inference_zero_shot_fast")
async def inference_zero_shot_fast(tts_text: str = Form()):
    model_output = cosyvoice.inference_zero_shot_fast(tts_text, stream=False)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--gpu_memory_utilization',
                        type=float,
                        default=0.1)
    parser.add_argument("--prompt_audio_path", default="./test_audio/zero_shot_prompt_laoban_16s_no_music.wav")
    parser.add_argument("--voice_type", type=int, default=0, help="0: laoban, 1:huawei, 2:xiong, 3:fast xiong, 4:laopo")

    args = parser.parse_args()

    assert args.voice_type in [0, 1, 2, 3, 4]
    voice_type2prompt = {
        # laoban, zero_shot_prompt_laoban_no_music.wav
        0: "第二届粤港澳大湾区博士、博士后创新创业大赛，在广州南沙这篇充满活力与机遇的沃土，向全球博士和博士后青年才俊们，发出诚挚的邀请，极目南沙，放眼世界，我们坚信，每一位参与大赛的青年才俊，都将在这里找到属于自己的舞台。",
        # huawei ren zheng fei, huawei_20second.wav
        1: "最重要是他们翅膀要硬，他们要自由去飞翔。这是父母的期望，父母并不是期望儿女来照顾父母，这个这个不是我们的期望。所以他们飞得越高，他们跟我们的差距就越大，代沟就越多，他愿意跟我们沟通就沟通，不愿意沟通我们就不沟通。",

        # xiong slow, xiong_hui_23second_slow.wav; 2 is better, 3 is too fast
        2: "现在我们有很多突出的矛盾，比如说人岗不匹配，比如说这个整个学科设置不合理，那么就整个会导致我们培养出来的学生的能力，和真正的市场需求，他是脱节的。那么这个问题为什么会产生呢，一方面是因为现在整个科技的发展在加速。",

        # xiong, xiong_hui_23second.wav
        3: "现在我们有很多突出的矛盾，比如说人岗不匹配，比如说这个整个学科设置不合理，那么就整个会导致我们培养出来的学生的能力，和真正的市场需求，他是脱节的。那么这个问题为什么会产生呢，一方面是因为现在整个科技的发展在加速，导致整个用工市场，对能力的需求的结构，也是在快速地变化。",
        # laopo, laopo2.wav
        4: "现在我们有很多突出的矛盾，比如说人岗不匹配，比如说这个整个学科设置不合理，那么就整个会导致我们培养出来的学生的能力，和真正的市场需求，他是脱节的。那么这个问题为什么会产生呢，一方面是因为现在整个科技的发展在加速，导致整个用工市场，对能力的需求的结构，也是在快速地变化。"
    }

    prompt_speech_text = voice_type2prompt[args.voice_type]
    prompt_speech_16k = load_wav(args.prompt_audio_path, 16000)


    try:
        cosyvoice = CosyVoice2(args.model_dir,
            load_jit=False, load_trt=False, load_vllm=True, fp16=True,
            prompt_text=prompt_speech_text, prompt_speech_16k=prompt_speech_16k,
            gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception:
        raise TypeError('failed to load cosyvoice2 model!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
