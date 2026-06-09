import requests
from dots_ocr.utils.image_utils import PILimage_to_base64
from openai import OpenAI
import os


def inference_with_vllm(
        image,
        prompt, 
        protocol="http",
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='rednote-hilab/dots.mocr',
        system_prompt=None,
        ):
    
    addr = f"{protocol}://{ip}:{port}/v1"
    client = OpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        resp = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError(f"vLLM trả về nội dung rỗng (model: {model_name})")
        return content
    except Exception as e:
        print(f"inference error: {e}")
        raise RuntimeError(f"Lỗi kết nối vLLM: {e}") from e

