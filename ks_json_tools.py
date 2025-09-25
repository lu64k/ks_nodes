import json
import random
import time
import os
import piexif
from typing import Any, Iterable
from PIL import Image
from .json_ultis import _parse_json_maybe_jsonl, parse_data, buildMetadata, process_exif_data

class KS_Json_Float_Range_Filter:
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "float_key": ("STRING", {"default": "click_rate", "multiline": False}),
                "min_val": ("FLOAT", {
                    "default": 0.180,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                }),
                "max_val": ("FLOAT", {
                    "default": 1.000,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_json",)
    FUNCTION = "filter_json"

    def filter_json(self, json_str, target_object, float_key, min_val, max_val):
        try:
            data = _parse_json_maybe_jsonl(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed with error: {str(e)}",)
        
        # 如果指定的 target_object 存在，则取出其值；否则直接使用 data
        subdata = parse_data(data, target_object)

        # 根据 subdata 的类型筛选数据
        if isinstance(subdata, dict):
            filtered = {}
            for k, item in subdata.items():
                if isinstance(item, dict) and float_key in item:
                    try:
                        value = float(item[float_key])
                    except Exception:
                        continue
                    if min_val <= value <= max_val:
                        filtered[k] = item
        elif isinstance(subdata, list):
            filtered = []
            for item in subdata:
                if isinstance(item, dict) and float_key in item:
                    try:
                        value = float(item[float_key])
                    except Exception:
                        continue
                    if min_val <= value <= max_val:
                        filtered.append(item)
        else:
            return ("Error: target_object data is neither a dict nor a list.",)

        # 如果 target_object 存在，则保持顶层键，否则直接输出过滤结果
        if target_object in data:
            result = {target_object: filtered}
        else:
            result = filtered

        return (json.dumps(result, indent=2, ensure_ascii=False),)

class KS_Json_Array_Constrains_Filter:
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "key_name": ("STRING", {"default": "prompt", "multiline": False}),
                "include_keywords": ("STRING", {"default": "", "multiline": False}),
                "exclude_keywords": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_json",)
    FUNCTION = "filter_json_by_keywords"

    def filter_json_by_keywords(self, json_str, target_object, key_name, include_keywords, exclude_keywords):
        """
        对输入的 JSON 数据进行筛选：
        1. 解析 json_str 得到数据结构；
        2. 如果顶层中存在 target_object，则对其对应的数据进行筛选（保持原有键名）；否则，将整个数据视为目标；
        3. 遍历目标中的每个对象，递归查找所有键名为 key_name 的值，将这些值合并成一个字符串（空格分隔）。
           - 若该字符串包含任一过滤词（不区分大小写），则跳过该对象；
           - 若提供筛选词，则仅保留合并字符串中至少包含一个筛选词的对象；
           - 如果两个关键词栏均为空，则输出原始数据。
        输出时：
           - 如果目标数据原来为字典，则保持原有键名；
           - 如果目标数据为列表，则输出列表。
        """
        try:
            data = _parse_json_maybe_jsonl(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed with error: {str(e)}",)
        
        include_list = [kw.strip() for kw in include_keywords.split(",") if kw.strip()] if include_keywords else []
        exclude_list = [kw.strip() for kw in exclude_keywords.split(",") if kw.strip()] if exclude_keywords else []

        # 判断是否存在 target_object，并处理
        if target_object in data:
            target_data = data[target_object]
            if isinstance(target_data, dict):
                filtered = {}
                for k, item in target_data.items():
                    found_values = self._recursive_find_key(item, key_name)
                    if not found_values:
                        continue
                    value_str = " ".join([str(v) for v in found_values])
                    value_lower = value_str.lower()
                    if exclude_list and any(ex_kw.lower() in value_lower for ex_kw in exclude_list):
                        continue
                    if include_list:
                        if any(inc_kw.lower() in value_lower for inc_kw in include_list):
                            filtered[k] = item
                    else:
                        filtered[k] = item
                # 如果两个关键词均为空，则直接保留原始数据
                if not include_list and not exclude_list:
                    filtered = target_data
                data[target_object] = filtered  # 保留顶层键
                result = data
            elif isinstance(target_data, list):
                filtered = []
                for item in target_data:
                    found_values = self._recursive_find_key(item, key_name)
                    if not found_values:
                        continue
                    value_str = " ".join([str(v) for v in found_values])
                    value_lower = value_str.lower()
                    if exclude_list and any(ex_kw.lower() in value_lower for ex_kw in exclude_list):
                        continue
                    if include_list:
                        if any(inc_kw.lower() in value_lower for inc_kw in include_list):
                            filtered.append(item)
                    else:
                        filtered.append(item)
                if not include_list and not exclude_list:
                    filtered = target_data
                data[target_object] = filtered  # 保留顶层键
                result = data
            else:
                return ("Error: target_object data is neither a dict nor a list.",)
        else:
            # 如果 target_object 不存在，则处理整个数据（转为列表）
            if isinstance(data, dict):
                target_data = list(data.values())
            elif isinstance(data, list):
                target_data = data
            else:
                return ("Error: Input JSON is neither a dict nor a list.",)
            filtered = []
            for item in target_data:
                found_values = self._recursive_find_key(item, key_name)
                if not found_values:
                    continue
                value_str = " ".join([str(v) for v in found_values])
                value_lower = value_str.lower()
                if exclude_list and any(ex_kw.lower() in value_lower for ex_kw in exclude_list):
                    continue
                if include_list:
                    if any(inc_kw.lower() in value_lower for inc_kw in include_list):
                        filtered.append(item)
                else:
                    filtered.append(item)
            if not include_list and not exclude_list:
                filtered = target_data
            result = filtered

        return (json.dumps(result, indent=2, ensure_ascii=False),)

    def _recursive_find_key(self, data, key_name):
        results = []
        if isinstance(data, dict):
            for k, v in data.items():
                if k == key_name:
                    results.append(v)
                if isinstance(v, (dict, list)):
                    results.extend(self._recursive_find_key(v, key_name))
        elif isinstance(data, list):
            for item in data:
                results.extend(self._recursive_find_key(item, key_name))
        return results

class KS_Json_Key_Replace_3ways:
    """
    节点名：json_key_replace
    功能：输入目标 JSON 字符串、键路径、键值和模式，对目标 JSON 中同名的键内容进行替换或追加。
    - 用户输入的键值格式为类似 "girl, boy, dogs"，节点会自动将其转换为数组形式：["girl", "boy", "dogs"]。
    - 键路径使用点号分隔来定位层级，例如 "subject.main_focus.main_objects" 或 "environment.background"。
    - 模式可选择 "replace"（替换）或 "append"（追加）。
    - 节点允许处理 4 个替换键，若留空则不做修改。
    """
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "key1_path": ("STRING", {"default": "", "multiline": False}),
                "key1_value": ("STRING", {"default": "", "multiline": False}),
                "key1_mode": (["replace", "append"], {"default": "replace"}),
                "key2_path": ("STRING", {"default": "", "multiline": False}),
                "key2_value": ("STRING", {"default": "", "multiline": False}),
                "key2_mode": (["replace", "append"], {"default": "replace"}),
                "key3_path": ("STRING", {"default": "", "multiline": False}),
                "key3_value": ("STRING", {"default": "", "multiline": False}),
                "key3_mode": (["replace", "append"], {"default": "replace"}),
                "key4_path": ("STRING", {"default": "", "multiline": False}),
                "key4_value": ("STRING", {"default": "", "multiline": False}),
                "key4_mode": (["replace", "append"], {"default": "replace"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_json",)
    FUNCTION = "json_key_replace"

    def json_key_replace(
        self,
        json_str,
        key1_path,
        key1_value,
        key1_mode,
        key2_path,
        key2_value,
        key2_mode,
        key3_path,
        key3_value,
        key3_mode,
        key4_path,
        key4_value,
        key4_mode
    ):
        # 解析输入 JSON
        try:
            data = _parse_json_maybe_jsonl(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed: {str(e)}",)

        # 辅助函数：将逗号分隔的字符串转换为数组
        def parse_key_value(s):
            s = s.strip()
            if not s:
                return None
            # 按逗号分割并去掉两侧空格
            return [item.strip() for item in s.split(",") if item.strip()]

        # 辅助函数：根据点号分隔的路径更新字典中对应键的值
        def update_dict_by_path(d, path, new_arr, mode):
            # 路径格式如 "subject.main_focus.main_objects"
            keys = path.split(".")
            current = d
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    # 若中间键不存在，则创建空字典
                    current[key] = {}
                current = current[key]
            last_key = keys[-1]
            if mode == "replace":
                current[last_key] = new_arr
            elif mode == "append":
                if last_key in current:
                    if isinstance(current[last_key], list):
                        current[last_key].extend(new_arr)
                    else:
                        current[last_key] = [current[last_key]] + new_arr
                else:
                    current[last_key] = new_arr

        # 依次处理 4 组替换键（若路径为空，则不修改）
        for key_path, key_value, key_mode in [
            (key1_path, key1_value, key1_mode),
            (key2_path, key2_value, key2_mode),
            (key3_path, key3_value, key3_mode),
            (key4_path, key4_value, key4_mode),
        ]:
            if key_path.strip():
                new_arr = parse_key_value(key_value)
                if new_arr is not None:
                    update_dict_by_path(data, key_path, new_arr, key_mode)

        modified_json_str = json.dumps(data, indent=2, ensure_ascii=False)
        return (modified_json_str,)

class KS_Json_Value_Eliminator:
    """
    节点名：json_value_eliminator
    功能：输入 JSON 字符串和剔除/筛选关键词，输出经过处理的 JSON 字符串。
         - 剔除关键词用逗号分隔，例如："boy, cat"。
         - 递归遍历 JSON 中所有最底层值（字符串），根据指定条件处理：
             * filter_mode=False（默认，剔除模式）：如果值中满足关键词条件，则删除该值。
             * filter_mode=True（包含模式）：如果值中不满足关键词条件，则删除该值。
         - logic_and 决定判断逻辑：
             * True：值必须同时包含所有关键词。
             * False：值只要包含任一关键词即可。
         - 保持原有 JSON 结构不变（只对值进行剔除）。
    """
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "eliminate_keywords": ("STRING", {"default": "", "multiline": False}),
                "filter_mode": ("BOOLEAN", {"default": False}),
                "logic_and": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_json",)
    FUNCTION = "json_value_eliminator"

    def json_value_eliminator(self, json_str, target_object, eliminate_keywords, filter_mode, logic_and):
        # 解析输入 JSON
        try:
            data = _parse_json_maybe_jsonl(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed: {str(e)}",)        
        try:
            # 使用 parse_data 辅助函数（需在模块顶层定义）处理数据格式
            data = parse_data(data, target_object)
        except Exception as e:
            return (str(e),)
        
        # 处理剔除/筛选关键词，转换为小写并去掉空格
        eliminate_list = [kw.strip().lower() for kw in eliminate_keywords.split(",") if kw.strip()]

        # 递归处理数据，根据 filter_mode 和 logic_and 进行处理
        self._eliminate_values(data, eliminate_list, filter_mode, logic_and)

        filtered_json_str = json.dumps(data, indent=2, ensure_ascii=False)
        return (filtered_json_str,)

    def _eliminate_values(self, data, eliminate_list, filter_mode, logic_and):
        """
        递归遍历 JSON 对象，处理每个最底层字符串值：
          - 如果 filter_mode 为 False（剔除模式），当值满足条件时将其置为 None。
          - 如果 filter_mode 为 True（包含模式），当值不满足条件时将其置为 None。
          逻辑判断使用 logic_and：
          - True：值必须包含所有关键词；
          - False：值只需包含任一关键词。
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self._eliminate_values(value, eliminate_list, filter_mode, logic_and)
                elif isinstance(value, str):
                    value_lower = value.lower()
                    if logic_and:
                        condition = all(kw in value_lower for kw in eliminate_list) if eliminate_list else False
                    else:
                        condition = any(kw in value_lower for kw in eliminate_list) if eliminate_list else False
                    # 根据模式决定是否删除该值
                    if filter_mode:
                        # 包含模式：如果值不满足条件，则删除
                        if not condition:
                            data[key] = None
                    else:
                        # 剔除模式：如果值满足条件，则删除
                        if condition:
                            data[key] = None
        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], (dict, list)):
                    self._eliminate_values(data[i], eliminate_list, filter_mode, logic_and)
                elif isinstance(data[i], str):
                    value_lower = data[i].lower()
                    if logic_and:
                        condition = all(kw in value_lower for kw in eliminate_list) if eliminate_list else False
                    else:
                        condition = any(kw in value_lower for kw in eliminate_list) if eliminate_list else False
                    if filter_mode:
                        if not condition:
                            data[i] = None
                    else:
                        if condition:
                            data[i] = None
            # 删除列表中为 None 的项
            data[:] = [item for item in data if item is not None]

class KS_Json_Extract_Key_And_Value_3ways:
    """
    节点名：extract_json_key_and_value
    功能：输入 JSON 字符串和最多五个键名，输出指定键名下的值。
         - 键名可以是单个键名（如 main_objects）或嵌套路径（如 subject.main_focus.main_objects）。
         - if_output_key 为 True 时，在输出中保留键名；否则只输出对应的值。
         - flatten 为 True 时，将嵌套列表扁平化后再输出，确保每个子项的抽取概率相等。
         ##2025/03/03##
    """
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "key1": ("STRING", {"default": "", "multiline": False}),
                "key2": ("STRING", {"default": "", "multiline": False}),
                "key3": ("STRING", {"default": "", "multiline": False}),
                "key4": ("STRING", {"default": "", "multiline": False}),
                "key5": ("STRING", {"default": "", "multiline": False}),
                "if_output_key": ("BOOLEAN", {"default": True}),
                "flatten": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = (
        "STRING",  # overall extracted
        "STRING",  # key1 extraction
        "STRING",  # key2 extraction
        "STRING",  # key3 extraction
        "STRING",  # key4 extraction
        "STRING",  # key5 extraction
    )
    RETURN_NAMES = (
        "extracted_json",
        "extracted_key1",
        "extracted_key2",
        "extracted_key3",
        "extracted_key4",
        "extracted_key5"
    )
    FUNCTION = "extract_json_key_and_value"

    def extract_json_key_and_value(self, json_str, target_object, key1, key2, key3, key4, key5, if_output_key, flatten):
        try:
            data = json.loads(json_str)
        except Exception as e:
            err = f"Error: JSON parsing failed: {str(e)}"
            return (err, err, err, err, err, err)
        
        try:
            # 使用 parse_data 辅助函数统一处理目标对象数据
            data = parse_data(data, target_object)
        except Exception as e:
            err = str(e)
            return (err, err, err, err, err, err)
        
        overall_extracted = []
        extracted_key1 = ""
        extracted_key2 = ""
        extracted_key3 = ""
        extracted_key4 = ""
        extracted_key5 = ""
        
        # 按顺序处理 5 个键
        keys = [key1, key2, key3, key4, key5]
        extracted_list = [None, None, None, None, None]
        for idx, key in enumerate(keys):
            if key.strip():
                value = self._get_value_by_path(data, key, flatten)
                if value is not None:
                    if if_output_key:
                        extracted_list[idx] = {key: value}
                    else:
                        if isinstance(value, list):
                            extracted_list[idx] = value
                        else:
                            extracted_list[idx] = value
                    overall_extracted.append({key: value} if if_output_key else value)
                else:
                    extracted_list[idx] = ""
            else:
                extracted_list[idx] = ""
        
        overall_extracted_json_str = json.dumps(overall_extracted, indent=2, ensure_ascii=False)
        extracted_key1 = json.dumps(extracted_list[0], indent=2, ensure_ascii=False)
        extracted_key2 = json.dumps(extracted_list[1], indent=2, ensure_ascii=False)
        extracted_key3 = json.dumps(extracted_list[2], indent=2, ensure_ascii=False)
        extracted_key4 = json.dumps(extracted_list[3], indent=2, ensure_ascii=False)
        extracted_key5 = json.dumps(extracted_list[4], indent=2, ensure_ascii=False)
        
        return (overall_extracted_json_str, extracted_key1, extracted_key2, extracted_key3, extracted_key4, extracted_key5)

    def _get_value_by_path(self, data, path, flatten):
        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                collected = []
                for item in current:
                    if isinstance(item, dict) and key in item:
                        collected.append(item[key])
                current = collected
            else:
                return None
        if flatten:
            current = self._flatten_list(current)
        else:
            if not isinstance(current, list):
                current = [current]
        return current

    def _flatten_list(self, lst):
        flat = []
        for item in lst:
            if isinstance(item, list):
                flat.extend(self._flatten_list(item))
            else:
                flat.append(item)
        return flat

class KS_Json_Key_Random_3ways:
    """
    节点名：json_key_random
    功能：从输入的 JSON 字符串中，根据用户指定的 3 个键名及对应的随机抽取数量，
          提取各键下的值（支持嵌套路径，使用点号分隔），并随机抽取指定数量的元素。
          如果键对应的值不是数组，则视为单元素数组。
          当 flatten 模式为 True 时，将所有对象的子项扁平化后再随机抽取，
          保证每个子项被抽中的概率相等。
          输出三个抽取结果，格式为 JSON 字符串。
          #已修改#2025/03/03
    """
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 10000000000}),
                "key1": ("STRING", {"default": "", "multiline": False}),
                "num1": ("INT", {"default": 1}),
                "key2": ("STRING", {"default": "", "multiline": False}),
                "num2": ("INT", {"default": 1}),
                "key3": ("STRING", {"default": "", "multiline": False}),
                "num3": ("INT", {"default": 1}),
                "min_val": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_val": ("INT", {"default": 1000000000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "flatten": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("result1", "result2", "result3")
    FUNCTION = "extract_random_keys"


    def extract_random_keys(self, json_str, target_object, seed, key1, num1, key2, num2, key3, num3, min_val, max_val, flatten):

        random.seed(seed)
        try:
            data = _parse_json_maybe_jsonl(json_str)
        except Exception as e:
            error = f"Error: JSON parsing failed: {str(e)}"
            return (error, error, error)
        
        # 如果指定的 target_object 存在，则使用其对应数据作为目标，否则用整个 JSON 数据
        if target_object in data:
            target_data = data[target_object]
            if isinstance(target_data, dict):
                data_list = list(target_data.values())
            elif isinstance(target_data, list):
                data_list = target_data
            else:
                return ("Error: target_object data is neither a dict nor a list.",)
        else:
            if isinstance(data, list):
                data_list = data
            elif isinstance(data, dict):
                data_list = list(data.values())
            else:
                return ("Error: Input JSON is neither a dict nor a list.",)
        
        result1 = self._extract_for_key(data_list, key1, num1, flatten)
        result2 = self._extract_for_key(data_list, key2, num2, flatten)
        result3 = self._extract_for_key(data_list, key3, num3, flatten)
        
        return (
            json.dumps(result1, ensure_ascii=False),
            json.dumps(result2, ensure_ascii=False),
            json.dumps(result3, ensure_ascii=False)
        )

    def _flatten_list(self, lst):
        """递归扁平化列表"""
        flat = []
        for item in lst:
            if isinstance(item, list):
                flat.extend(self._flatten_list(item))
            else:
                flat.append(item)
        return flat

    def _extract_for_key(self, data_list, key, num, flatten):
        # 如果 key 为空，则直接返回 data_list
        if not key.strip():
            # 如果 flatten 为 True 且 data_list嵌套，需要扁平化
            if flatten:
                return self._flatten_list(data_list)
            else:
                return data_list

        # 否则根据点号分割路径
        keys = key.split(".")
        current = data_list
        for k in keys:
            collected = []
            if isinstance(current, list):
                for item in current:
                    # 如果 item 是字典并且包含 k，则提取；如果 item 不是字典，则忽略
                    if isinstance(item, dict) and k in item:
                        collected.append(item[k])
                current = collected
            else:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return None
        
        # 如果 flatten 模式开启且当前结果为嵌套列表，则扁平化
        if flatten:
            current = self._flatten_list(current)
        else:
            if not isinstance(current, list):
                current = [current]
        
        if len(current) == 0:
            return []
        if num >= len(current):
            return current
        return random.sample(current, num)

class KS_Json_Count:
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_json_items"

    def count_json_items(self, json_str, target_object):
        try:
            data = _parse_json_maybe_jsonl(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed: {str(e)}",)
        
        # 如果顶层存在 target_object，则使用其对应的数据，否则直接使用整个 JSON 数据
        if isinstance(data, dict) and target_object in data:
            target_data = data[target_object]
        else:
            target_data = data

        if isinstance(target_data, dict):
            count = len(target_data)
        elif isinstance(target_data, list):
            count = len(target_data)
        else:
            count = 0
        print (count)
        return (count,)

class KS_JsonToString:
    CATEGORY = "ksjson_nodes/tools"

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("JSON", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_str",)
    FUNCTION = "to_string"

    def to_string(self, data):
        """
        1. 将 Python 对象序列化为带缩进的 JSON 文本；
        2. 对其中的转义序列（如 \n, \"）解码为实际字符，保留中文字符；
        3. 如果文本最外层有多余的引号，则去除。
        """
        # 步骤 1: 序列化
        try:
            raw = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            raw = repr(data)

        # 步骤 2: 解码转义序列，保留中文
        try:
            # 直接用 str.encode().decode('unicode_escape') 会破坏中文
            # 改用 json.loads 重新解析并序列化，确保转义正确处理
            parsed = json.loads(raw)  # 解析回Python对象
            text = json.dumps(parsed, ensure_ascii=False, indent=2)  # 重新序列化
        except Exception:
            text = raw

        # 步骤 3: 去除最外层引号
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        return (text,)

class KS_JsonKeyReplacer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False
                }),
                "keyname": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "new_value": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_json",)
    FUNCTION = "replace_key"
    CATEGORY = "ksjson_nodes/tools"

    def _find_and_replace_key(self, obj, keyname, new_value, key_count, key_paths):
        """
        递归遍历 JSON 对象，查找并替换指定键的值，同时记录键的出现次数和路径。
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == keyname:
                    key_count[0] += 1
                    key_paths.append(f"{key_paths[-1]}.{key}" if key_paths else key)
                    obj[key] = new_value
                self._find_and_replace_key(value, keyname, new_value, key_count, key_paths)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._find_and_replace_key(item, keyname, new_value, key_count, key_paths + [f"[{i}]"])

    def replace_key(self, json_string: str, keyname: str, new_value: str):
        # 检查输入是否有效
        if not json_string.strip():
            return ("Error: JSON string is empty",)
        if not keyname.strip():
            return ("Error: Keyname is empty",)

        # 尝试解析 JSON 字符串
        try:
            json_obj = json.loads(json_string)
        except json.JSONDecodeError:
            return ("Error: Invalid JSON string",)

        # 尝试解析 new_value 作为 JSON（支持数字、字符串等）
        try:
            parsed_new_value = json.loads(new_value)
        except json.JSONDecodeError:
            parsed_new_value = new_value  # 如果无法解析，保持为字符串

        # 统计键的出现次数并替换
        key_count = [0]  # 用列表记录计数以便在递归中修改
        key_paths = []   # 记录键的路径
        self._find_and_replace_key(json_obj, keyname, parsed_new_value, key_count, key_paths)

        # 检查键的唯一性
        if key_count[0] == 0:
            return (f"Error: Key '{keyname}' not found in JSON",)
        if key_count[0] > 1:
            return (f"Error: Key '{keyname}' is not unique, found at paths: {', '.join(key_paths)}",)

        # 将修改后的 JSON 对象转回字符串
        modified_json = json.dumps(json_obj, ensure_ascii=False)
        return (modified_json,)

class KS_JsonKeyExtractor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False
                }),
                "keyname": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "keep_key": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_json",)
    FUNCTION = "extract_key"
    CATEGORY = "ksjson_nodes/tools"

    def _find_key(self, obj, keyname, results, key_paths, current_path=""):
        """
        递归遍历 JSON 对象，查找指定键的值，记录路径和结果。
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if key == keyname:
                    results.append(value)
                    key_paths.append(new_path)
                self._find_key(value, keyname, results, key_paths, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]"
                self._find_key(item, keyname, results, key_paths, new_path)

    def extract_key(self, json_string: str, keyname: str, keep_key: bool):
        # 检查输入是否有效
        if not json_string.strip():
            return ("Error: JSON string is empty",)

        # 尝试解析 JSON 字符串
        try:
            json_obj = json.loads(json_string)
        except json.JSONDecodeError:
            return ("Error: Invalid JSON string",)

        # 如果 keyname 为空，返回顶层 JSON
        if not keyname.strip():
            if keep_key:
                return (json.dumps(json_obj, ensure_ascii=False),)
            else:
                return (json.dumps(json_obj, ensure_ascii=False),)  # 顶层已经是对象

        # 查找指定键
        results = []
        key_paths = []
        self._find_key(json_obj, keyname, results, key_paths)

        # 检查键的唯一性
        if not results:
            return (f"Error: Key '{keyname}' not found in JSON",)
        if len(results) > 1:
            return (f"Error: Key '{keyname}' is not unique, found at paths: {', '.join(key_paths)}",)

        # 根据 keep_key 返回键值对或仅值
        if keep_key:
            result = {keyname: results[0]}
        else:
            result = results[0]

        # 转回 JSON 字符串
        return (json.dumps(result, ensure_ascii=False),)

class KS_JsonlFolderMatchReader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jsonl_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "target_key": ("STRING", {
                    "default": "uid",
                    "multiline": False
                }),
                "file_extension": ("STRING", {
                    "default": ".png",
                    "multiline": False
                }),
                "folder_limit": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 10000,
                    "step": 1
                }),
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999999,
                    "step": 1
                }),
                "random_order": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_string",)
    FUNCTION = "read_jsonl_folder_match"
    CATEGORY = "ksjson_nodes/tools"

    def read_jsonl_folder_match(self, jsonl_path: str, folder_path: str, target_key: str, file_extension: str, folder_limit: int, random_seed: int, random_order: bool):
        # 调试：打印所有输入
        print(f"输入参数 - jsonl_path: {jsonl_path}, folder_path: {folder_path}, target_key: {target_key}, file_extension: {file_extension}, folder_limit: {folder_limit}, random_seed: {random_seed}, random_order: {random_order}")

        # 检查文件夹路径
        if not os.path.isdir(folder_path):
            raise Exception(f"Folder {folder_path} does not exist or is not a directory")

        # 强制刷新文件夹文件列表
        try:
            time.sleep(0.1)  # 短暂等待文件系统同步
            folder_files = []
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    if f.endswith(file_extension):  # 过滤指定扩展名
                        # 最小改动：只保存文件名，保持与原先 os.listdir 的输出格式一致
                        folder_files.append(f)

            print(f"当前文件夹及子文件夹的文件: {folder_files}")
        except Exception as e:
            raise Exception(f"Error accessing folder {folder_path}: {str(e)}")

        # 检查文件夹文件数是否超过上限
        if len(folder_files) > folder_limit:
            raise Exception(f"Folder contains {len(folder_files)} files, exceeding limit of {folder_limit}")

        # 去掉文件后缀，用于比对
        folder_file_basenames = [os.path.splitext(f)[0] for f in folder_files]
        print(f"去后缀后的文件名: {folder_file_basenames}")

        # 读取 jsonl 文件，提取 target_key 的值
        jsonl_values = []
        if os.path.exists(jsonl_path) and os.path.isfile(jsonl_path):
            # 与原逻辑一致：从 .jsonl 文件读取
            if not jsonl_path.endswith(".jsonl"):
                raise Exception(f"File {jsonl_path} is not a .jsonl file")
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        try:
                            json_obj = json.loads(line.strip())
                            if not isinstance(json_obj, dict):
                                raise Exception(f"Line {i} is not a JSON object")
                            jsonl_values.append(json_obj)
                        except json.JSONDecodeError:
                            raise Exception(f"Invalid JSON at line {i} in {jsonl_path}")
            except Exception as e:
                raise Exception(f"Error reading JSONL file: {str(e)}")
        else:
            # 新增：将 jsonl_path 当作 JSON/JSONL 内容解析
            jsonl_values = _parse_json_maybe_jsonl(jsonl_path)

        # 检查文件夹里的文件（去后缀后）是否都在 jsonl_values 里
        for file_basename in folder_file_basenames:
            if file_basename not in jsonl_values:
                print("some files is not include in the jsonl list")

        # 按 random_order 决定读取方式
        if random_order:
            # 收集所有未处理的 jsonl 条目
            try:
#                with open(jsonl_path, 'r', encoding='utf-8') as f:
                unprocessed_entries = []
                for i, json_obj in enumerate(jsonl_values, 1):
                    if isinstance(json_obj, dict) and target_key in json_obj and json_obj[target_key] not in folder_file_basenames:
                        unprocessed_entries.append(json_obj)
            except Exception as e:
                raise Exception(f"Error reading JSONL file: {str(e)}")

            # 如果没有未处理的条目，抛异常
            if not unprocessed_entries:
                raise Exception(f"All JSONL entries with key {target_key} already exist in folder")

            # 随机选一条
            selected_entry = random.choice(unprocessed_entries)
            json_string = json.dumps(selected_entry, ensure_ascii=False)
        else:
            # 顺序读取（旧版逻辑）：逐行找第一个未处理的
            try:
#                with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, json_obj in enumerate(jsonl_values, 1):
                    if isinstance(json_obj, dict) and target_key in json_obj and json_obj[target_key] not in folder_file_basenames:
                        json_string = json.dumps(json_obj, ensure_ascii=False)
                        print(f"返回 JSON: {json_string}")
                        return (json_string,)
                raise Exception(f"All JSONL entries with key {target_key} already exist in folder")
            except Exception as e:
                raise Exception(f"Error reading JSONL file: {str(e)}")

        print(f"返回 JSON: {json_string}")
        return (json_string,)

class KS_Json_loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_list_str": ("STRING", {
                    "default": "input json_str or path to the jsonfile(txt, json, jsonl)",
                    "multiline": True
                }),
                "start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999999,
                    "step": 1
                }),
                "count": ("INT", {
                    "default": -1,   # -1 表示到末尾
                    "min": -1,
                    "max": 999999999,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_list_str",)
    FUNCTION = "slice_json_list_str"
    CATEGORY = "ksjson_nodes/tools"

    def slice_json_list_str(self, json_list_str: str, start: int, count: int):
        items = _parse_json_maybe_jsonl(json_list_str)  # 也兼容 JSONL 输入
        n = len(items)
        end = start + count
        if count < 0 or end > n:
            end = n
        if start < 0 or start > end:
            raise Exception(f"Invalid range: start={start}, end={end}, total={n}")
        sliced = items[start:end]
        return (json.dumps(sliced, ensure_ascii=False),)

class KS_make_json_node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key1": ("STRING", {"default": "", "multiline": False}),
                "value1": ("STRING", {"default": "", "multiline": True}),
                "key2": ("STRING", {"default": "", "multiline": False}),
                "value2": ("STRING", {"default": "", "multiline": True}),
                "key3": ("STRING", {"default": "", "multiline": False}),
                "value3": ("STRING", {"default": "", "multiline": True}),
                "key4": ("STRING", {"default": "", "multiline": False}),
                "value4": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("json1", "json2", "json3", "json4",)
    FUNCTION = "make_json"
    CATEGORY = "Sikai_JSON"

    def _parse_value(self, value):
        """尝试解析value为适当类型（整数、布尔值、JSON对象或字符串）"""
        value = value.strip()  # 去掉首尾空白和换行
        if not value:
            return None
        # 尝试解析为布尔值
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        # 尝试解析为整数
        try:
            return int(value)
        except ValueError:
            pass
        # 尝试解析为JSON（比如嵌套对象）
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        # 默认当字符串
        return value

    def make_json(self, key1, value1, key2, value2, key3, value3, key4, value4):
        json_outputs = ["", "", "", ""]

        # 构造键值对
        pairs = [(key1, value1), (key2, value2), (key3, value3), (key4, value4)]
        for i, (key, value) in enumerate(pairs):
            if key and value:  # 只有键值都不为空才生成JSON
                try:
                    parsed_value = self._parse_value(value)
                    if parsed_value is not None:
                        json_dict = {key: parsed_value}
                        json_outputs[i] = json.dumps(json_dict, ensure_ascii=False)
                except Exception as e:
                    print(f"键值对 {key}:{value} 转JSON失败: {str(e)}")
                    json_outputs[i] = ""

        print(f"生成的JSON: {json_outputs}")
        return json_outputs

class KS_merge_json_node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json1": ("STRING", {"default": "", "multiline": False}),
                "json2": ("STRING", {"default": "", "multiline": False}),
                "json3": ("STRING", {"default": "", "multiline": False}),
                "json4": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_str",)
    FUNCTION = "merge_json"
    CATEGORY = "Sikai_JSON"

    def merge_json(self, json1, json2, json3, json4):
        merged_dict = {}

        # 处理每个JSON输入
        for i, json_str in enumerate([json1, json2, json3, json4], 1):
            if json_str:  # 跳过空字符串
                try:
                    json_dict = json.loads(json_str)
                    merged_dict.update(json_dict)
                except json.JSONDecodeError as e:
                    print(f"JSON{i} 解析失败: {str(e)}, 输入: {json_str}")
                    continue

        # 转成JSON字符串
        try:
            json_str = json.dumps(merged_dict)
            print(f"合并后的JSON: {json_str}")
            return (json_str,)
        except Exception as e:
            print(f"合并JSON失败: {str(e)}")
            return ("",)

class KS_image_metadata_node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False})
            },
        }

    RETURN_TYPES = ("JSON","JSON","JSON")
    RETURN_NAMES = ("prompt", "metadata", "workflow")
    FUNCTION = "extract_metadata"
    CATEGORY = "Sikai_JSON"

    def extract_metadata(self,image_path):
        """
        1) Calls your buildMetadata helper (which must return something like
        (img_obj, prompt_dict, metadata_dict))
        2) If the image is WEBP, tries to pull real EXIF with piexif and re-compute
        prompt/metadata via process_exif_data.
        3) Returns (prompt_dict, metadata_dict).  
        """
        # 1) Open image so you can check format
        img = Image.open(image_path)
        
        # 2) Call your existing helper to get a baseline prompt/metadata
        #    (we ignore the returned img_obj here, since we just needed its prompt/metadata)
        img, prompt, metadata, workflow = buildMetadata(image_path)
        
        # 3) If it’s a WEBP, override with true EXIF if possible
        if img.format == "WEBP":
            try:
                exif_dict = piexif.load(image_path)
                # process_exif_data should be a free function or imported at top;
                # if it’s a method on a class, call it on an instance, not via `self`
                prompt, metadata = process_exif_data(exif_dict)
            except (piexif.InvalidImageDataError, ValueError, KeyError) as e:
                # failed to parse EXIF—fall back to empty or leave original
                prompt = {}
                metadata = {}
                workflow = {}

        return prompt, metadata, workflow


class KS_Save_JSON:
    """
    将 json_str 保存为 jsonl / json / txt 文件的 ComfyUI 节点。
    - JSONL: list -> 每元素一行；dict -> 每个 (k, v) 一行，写入 {"key": k, "value": v}
    - JSON : 整体写入，可选择是否 pretty（缩进）
    - TXT  : 原样写入字符串
    """
    CATEGORY = "Sikai_nodes/tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "./output.jsonl", "multiline": False}),
                "json_str": ("STRING", {"default": "{}", "multiline": True}),
                "save_mode": (["overwrite", "append", "new only"],),
                "save_format": (["jsonl", "json", "txt"],),
                "pretty": ("BOOLEAN", {"default": True}),  # 仅对 JSON 格式生效
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_data"

    # ---- 内部工具 ----
    def _ensure_parent_dir(self, path: str):
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def _parse_json_if_needed(self, json_str: str, save_format: str) -> Any:
        """
        txt 格式不解析，json/jsonl 需要解析为 Python 对象
        """
        if save_format == "txt":
            return json_str
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")

    def _open_mode(self, save_mode: str) -> str:
        if save_mode == "overwrite":
            return "w"
        elif save_mode == "append":
            return "a"
        elif save_mode == "new only":
            # 调用方会先判断是否存在，这里按写新文件方式打开
            return "x"
        else:
            # 兜底，用覆盖
            return "w"

    def _jsonl_iter_lines(self, data: Any) -> Iterable[str]:
        """
        将 list/dict/标量 转为逐行 JSON 文本（不带换行符，调用处追加 '\n'）
        - list: 每个元素一行
        - dict: 每个键值对一行，格式 {"key": k, "value": v}
        - 其他: 直接一行
        """
        if isinstance(data, list):
            for item in data:
                yield json.dumps(item, ensure_ascii=False)
        elif isinstance(data, dict):
            for k, v in data.items():
                yield json.dumps({"key": k, "value": v}, ensure_ascii=False)
        else:
            # 标量或其他结构，整体一行
            yield json.dumps(data, ensure_ascii=False)

    # ---- 主逻辑 ----
    def save_data(self, file_path: str, json_str: str, save_mode: str, save_format: str, pretty: bool):
        """
        将 json_str 保存为 jsonl / json / txt
        """
        try:
            # 解析 or 直写
            payload = self._parse_json_if_needed(json_str, save_format)

            # new only 检查
            if save_mode == "new only" and os.path.exists(file_path):
                return (f"File '{file_path}' already exists. No changes made.",)

            self._ensure_parent_dir(file_path)

            # 写入
            if save_format == "txt":
                # 原样文本
                mode = self._open_mode(save_mode)
                with open(file_path, mode, encoding="utf-8") as f:
                    f.write(payload)
                return (f"TXT saved to '{file_path}' with mode '{save_mode}'.",)

            elif save_format == "json":
                mode = self._open_mode(save_mode)
                with open(file_path, mode, encoding="utf-8") as f:
                    if pretty:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                        f.write("\n")  # 末尾换行更友好
                    else:
                        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                return (f"JSON saved to '{file_path}' with mode '{save_mode}', pretty={pretty}.",)

            elif save_format == "jsonl":
                # 每条记录写一行，末尾加 '\n'
                mode = self._open_mode(save_mode)
                with open(file_path, mode, encoding="utf-8") as f:
                    for line in self._jsonl_iter_lines(payload):
                        f.write(line + "\n")
                return (f"JSONL saved to '{file_path}' with mode '{save_mode}'.",)

            else:
                return (f"Unsupported format: {save_format}",)

        except FileExistsError:
            # 来自 new only 模式的 'x' 打开
            return (f"File '{file_path}' already exists. No changes made.",)
        except Exception as e:
            return (f"Error: {e}",)