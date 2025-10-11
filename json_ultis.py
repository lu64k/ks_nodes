import json
import random
import torch
import time
import datetime
from datetime import datetime
import re
import os
import piexif
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from PIL.PngImagePlugin import PngImageFile
from PIL.JpegImagePlugin import JpegImageFile


def buildMetadata(image_path):
    if not Path(image_path).is_file():
        # 直接抛 FileNotFoundError，提示里带上路径即可
        raise FileNotFoundError(f"找不到文件：{image_path}")

    img = Image.open(image_path)

    metadata = {}
    prompt = {}
    workflow= {}

    metadata["fileinfo"] = {
        "filename": Path(image_path).as_posix(),
        "resolution": f"{img.width}x{img.height}",
        "date": str(datetime.fromtimestamp(os.path.getmtime(image_path))),
    }

    # only for png files
    if isinstance(img, PngImageFile):
        metadataFromImg = img.info
        for key, value in metadataFromImg.items():
            if isinstance(value, bytes):
                try:
                    metadataFromImg[key] = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Failed to decode {key}: {e}")

        # for all metadataFromImg convert to string (but not for workflow and prompt!)
        for k, v in metadataFromImg.items():
            # from ComfyUI
            if k == "workflow":
                try:
                    metadata["workflow"] = json.loads(metadataFromImg["workflow"])
                    workflow = metadata["workflow"]
                except Exception as e:
                    print(f"Error parsing metadataFromImg 'workflow': {e}")

            # from ComfyUI
            elif k == "prompt":
                try:
                    metadata["prompt"] = json.loads(metadataFromImg["prompt"])
                    
                    prompt = metadata["prompt"]
                    
                except Exception as e:
                    print(f"Error parsing metadataFromImg 'prompt': {e}")

            else:
                try:
                    # for all possible metadataFromImg by user
                    metadata[str(k)] = json.loads(v)
                except Exception as e:
                    print(f"Error parsing {k} as json, trying as string: {e}")
                    try:
                        metadata[str(k)] = str(v)
                    except Exception as e:
                        print(f"Error parsing {k} it will be skipped: {e}")

    if isinstance(img, JpegImageFile):
        exif = img.getexif()

        for k, v in exif.items():
            tag = TAGS.get(k, k)
            if v is not None:
                metadata[str(tag)] = str(v)

        for ifd_id in IFD:
            try:
                if ifd_id == IFD.GPSInfo:
                    resolve = GPSTAGS
                else:
                    resolve = TAGS

                ifd = exif.get_ifd(ifd_id)
                ifd_name = str(ifd_id.name)
                metadata[ifd_name] = {}

                for k, v in ifd.items():
                    tag = resolve.get(k, k)
                    metadata[ifd_name][str(tag)] = str(v)

            except KeyError:
                pass
    return img, prompt, metadata, workflow

def process_exif_data(exif_data):
    metadata = {}
    # 检查 '0th' 键下的 271 值，提取 Prompt 信息
    if '0th' in exif_data and 271 in exif_data['0th']:
        prompt_data = exif_data['0th'][271].decode('utf-8')
        # 移除可能的前缀 'Prompt:'
        prompt_data = prompt_data.replace('Prompt:', '', 1)
        # 假设 prompt_data 是一个字符串，尝试将其转换为 JSON 对象
        try:
            metadata['prompt'] = json.loads(prompt_data)
        except json.JSONDecodeError:
            metadata['prompt'] = prompt_data

    # 检查 '0th' 键下的 270 值，提取 Workflow 信息
    if '0th' in exif_data and 270 in exif_data['0th']:
        workflow_data = exif_data['0th'][270].decode('utf-8')
        # 移除可能的前缀 'Workflow:'
        workflow_data = workflow_data.replace('Workflow:', '', 1)
        try:
            # 尝试将字节字符串转换为 JSON 对象
            metadata['workflow'] = json.loads(workflow_data)
        except json.JSONDecodeError:
            # 如果转换失败，则将原始字符串存储在 metadata 中
            metadata['workflow'] = workflow_data

    metadata.update(exif_data)
    return metadata

def read_jsonl_to_list_str(jsonl_path: str):
    if not os.path.exists(jsonl_path) or not os.path.isfile(jsonl_path):
        raise Exception(f"JSONL file {jsonl_path} does not exist")
    if not jsonl_path.endswith(".jsonl"):
        raise Exception(f"File {jsonl_path} is not a .jsonl file")

    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON at line {i} in {jsonl_path}")
            if not isinstance(obj, dict):
                raise Exception(f"Line {i} is not a JSON object")
            items.append(obj)
    return items

def _parse_json_maybe_jsonl(s: str) -> list[dict]:
    """
    支持三种输入：
    1) JSON 数组字符串:   "[{...},{...}]"
    2) JSON 对象字符串:   "{...}"          -> 视为单元素列表
    3) JSONL 多行字符串:  每行一个 JSON 对象
    4) 文件路径: 支持 .txt, .json, .jsonl 文件，包含合法 JSON 内容
    返回: list[dict]
    支持中文字符的编码格式（默认 UTF-8）。
    """
    # 确保输入是字符串并去除首尾空白
    if not isinstance(s, str):
        raise TypeError("输入必须是字符串类型")
    
    s = (s or "").strip()
    if not s:
        return []

    # 1. 判断是否是文件路径
    if os.path.exists(s) and os.path.isfile(s):
        if not s.lower().endswith(('.txt', '.json', '.jsonl')):
            raise ValueError(f"文件 {s} 扩展名不支持，仅支持 .txt, .json, .jsonl")
        try:
            with open(s, 'r', encoding='utf-8') as f:
                s = f.read().strip()
        except Exception as e:
            raise IOError(f"读取文件 {s} 失败: {str(e)}")
        
    # 统一处理编码
    if isinstance(s, (bytes, bytearray)):
        s = s.decode('utf-8', errors='replace')

    # JSON 数组
    if s.startswith("["):
        try:
            data = json.loads(s)
            if not isinstance(data, list):
                raise Exception("输入的字符串不是 JSON 数组")
            return data
        except json.JSONDecodeError as e:
            raise Exception(f"解析 JSON 数组失败: {str(e)}")

    # 3. 尝试作为 JSONL 解析
    if "\n" in s:
        try:
            out = []
            for i, line in enumerate(s.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"第 {i} 行不是合法 JSON 对象")
                out.append(obj)
            if out:  # 如果成功解析出至少一个对象
                return out
        except json.JSONDecodeError:
            pass  # JSONL 解析失败，继续尝试作为单个 JSON 对象解析

    # 4. 尝试作为单个 JSON 对象解析
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("输入的 JSON 不是对象")
        return [obj]
    except json.JSONDecodeError as e:
        raise ValueError(f"解析 JSON 对象失败: {str(e)}")
    
def parse_data(data, target_object):
    """
    根据 target_object 自动处理 JSON 数据：
    - 如果 data 中存在 target_object，则取其值；若是 dict，则转换为列表（取所有值）；若是列表，则直接使用。
    - 如果 data 中不存在 target_object，则：
         - 如果 data 是列表，则直接返回；
         - 如果 data 是 dict，则返回所有值组成的列表。
    """
    if target_object in data:
        target_data = data[target_object]
        if isinstance(target_data, dict):
            return list(target_data.values())
        elif isinstance(target_data, list):
            return target_data
        else:
            raise ValueError("Error: target_object data is neither a dict nor a list.")
    else:
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
        else:
            raise ValueError("Error: Input JSON is neither a dict nor a list.")

