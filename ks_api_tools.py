import tenacity
import requests
from http import HTTPStatus
import json
import base64
#from fastapi import HTTPException

def handle_response(response, seed):
    try:
        # 检查Content-Type
        content_type = response.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            # JSON响应
            response_data = response.json()
            if not isinstance(response_data, dict) or 'data' not in response_data or not response_data['data']:
                raise ValueError("响应中缺少data字段或data为空")
            
            data = response_data['data'][0]
            if not isinstance(data, dict):
                raise ValueError("data[0]不是有效字典")
                
            if 'b64_json' in data:
                # 加上data URI前缀，兼容get_base64_image
                b64_with_prefix = f"data:image/png;base64,{data['b64_json']}"
                print(f"成功拿到b64_json: {data['b64_json'][:50]}...")
                return (b64_with_prefix, seed)
            elif 'url' in data:
                print(f"成功拿到URL: {data['url']}")
                return (data['url'], seed)
            else:
                raise ValueError("data中缺少b64_json或url字段")
        
        elif "image/png" in content_type:
            # 二进制图片响应，转base64并加前缀
            img_data = response.content
            if not img_data:
                raise ValueError("图片数据为空")
            b64_data = base64.b64encode(img_data).decode("utf-8")
            b64_with_prefix = f"data:image/png;base64,{b64_data}"
            print(f"成功拿到图片并转为b64_json: {b64_data[:50]}...")
            return (b64_with_prefix, seed)
        
        else:
            raise ValueError(f"不支持的Content-Type: {content_type}")
    
    except:
        print("error")
    
class KS_any_payload_image_API_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "payload": ("STRING", {"defaultInput": True, "default": "{}"}),
                "headers": ("STRING", {"defaultInput": True, "default": "{}"}),
                "api_url": ("STRING", {"multiline": False, "default": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("b64_or_url", "seed",)
    FUNCTION = "get_image"
    CATEGORY = "Sikai_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1.25, min=5, max=30), stop=tenacity.stop_after_attempt(3))
    def get_image(self, payload, headers, api_url, seed):
        try:
            # 调试输入
            print(f"input payload: {payload}")
            print(f"input headers: {headers}")

            # 解析JSON字符串
            try:
                payload_dict = json.loads(payload) if payload else {}
            except json.JSONDecodeError as e:
                raise Exception(f"failed to parse Payload JSON: {str(e)}")
            
            try:
                headers_dict = json.loads(headers) if headers else {}
            except json.JSONDecodeError as e:
                raise Exception(f"failed to parse Headers JSON: {str(e)}")

            # 发送POST请求
            response = requests.post(api_url, json=payload_dict, headers=headers_dict)
            print(f"status code: {response.status_code}")
            print(f"call respond: {response.text}")

            if response.status_code != HTTPStatus.OK:
                try:
                    response_data = response.json()
                    raise Exception(f"failed to call API - Code: {response_data.get('code', 'unknow')}, Message: {response_data.get('message', 'unknow')}")
                except ValueError:
                    raise Exception(f"failed to call API - unable to parse the response: {response.text}")

            # 解析响应
            img_data, seed = handle_response(response, seed)
            return img_data, seed

        except Exception as e:
            print(f"error: {str(e)}")
            raise