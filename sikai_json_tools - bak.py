import json
import random
import torch
import time
import datetime
from datetime import datetime
import re
import nltk
from nltk import pos_tag, word_tokenize
# 如果未下载过模型，可以取消下一行注释
nltk.download('averaged_perceptron_tagger')



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


class SK_Json_Float_Range_Filter:
    CATEGORY = "Sikai_nodes/tools"

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
            data = json.loads(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed with error: {str(e)}",)
        
        # 如果指定的 target_object 存在，则取出其值；否则直接使用 data
        if target_object in data:
            subdata = data[target_object]
        else:
            subdata = data

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

class SK_Json_Array_Constrains_Filter:
    CATEGORY = "Sikai_nodes/tools"

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
            data = json.loads(json_str)
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

class SK_Json_Merge_Extract:
    """
    1. 解析输入的 JSON 数组并合并为一个对象（同层键合并为列表）。
    2. 针对以下 9 个数组字段，执行随机抽取（可选固定随机种子），再组装回一个新的单个 JSON 对象：
       - subject->main_focus->main_objects
       - subject->main_focus->main_details
       - subject->main_focus->actions (命名为 subject_actions)
       - environment->background (若原为字符串，则视为单元素列表)
       - environment->environment_elements
       - environment->actions (命名为 environment_actions)
       - style
       - lighting_and_atmosphere
       - others
    3. 输出 10 个字符串：
       1) final_image_json         (完整结构的 JSON)
       2) main_objects_str
       3) main_details_str
       4) subject_actions_str
       5) background_str
       6) environment_elements_str
       7) environment_actions_str
       8) style_str
       9) lighting_and_atmosphere_str
       10) others_str
    """
    CATEGORY = "Sikai_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "fix_random": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 10000000000}),
                "num_main_objects": ("INT", {"default": 2}),
                "num_main_details": ("INT", {"default": 2}),
                "num_main_actions": ("INT", {"default": 1}),
                "num_background": ("INT", {"default": 1}),
            
                "num_environment_elements": ("INT", {"default": 2}),
                "num_environment_actions": ("INT", {"default": 1}),
                "num_style": ("INT", {"default": 2}),
                "num_lighting_atmosphere": ("INT", {"default": 2}),
                "num_others": ("INT", {"default": 2}),
            }
        }

    RETURN_TYPES = (
        "STRING",  # final_image_json
        "STRING",  # main_objects_str
        "STRING",  # main_details_str
        "STRING",  # main_actions_str
        "STRING",  # background_str
        "STRING",  # environment_elements_str
        "STRING",  # environment_actions_str
        "STRING",  # style_str
        "STRING",  # lighting_and_atmosphere_str
        "STRING",  # others_str
    )

    RETURN_NAMES = (
        "final_image_json",
        "main_objects_str",
        "main_details_str",
        "main_actions_str",
        "background_str",
        "environment_elements_str",
        "environment_actions_str",
        "style_str",
        "lighting_and_atmosphere_str",
        "others_str"
    )

    FUNCTION = "merge_and_extract"

    def merge_and_extract(
        self,
        json_str,
        target_object,
        fix_random,
        seed,
        num_main_objects,
        num_main_details,
        num_main_actions,
        num_background,
        num_environment_elements,
        num_environment_actions,
        num_style,
        num_lighting_atmosphere,
        num_others
    ):
        # 设置随机种子：若 fix_random 为 True，使用用户提供的 seed；否则，使用当前时间戳（秒）作为种子
        if fix_random:
            random.seed(seed)
        else:
            random.seed(int(time.time()))

        # 1. 解析输入的 JSON 数组
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

        # 2. 合并所有 JSON 对象
        merged = self._merge_json_objects(data)

        # 3. 针对目标字段随机抽取
        final_obj = {
            "subject": {
                "main_focus": {}
            },
            "environment": {},
            "style": [],
            "lighting_and_atmosphere": [],
            "others": []
        }

        # ========== subject->main_focus->main_objects ==========
        main_objects_full = self._get_as_list(merged, ["subject", "main_focus", "main_objects"])
        main_objects_pick = self._random_pick(main_objects_full, num_main_objects)
        final_obj["subject"]["main_focus"]["main_objects"] = main_objects_pick

        # ========== subject->main_focus->main_details ==========
        main_details_full = self._get_as_list(merged, ["subject", "main_focus", "main_details"])
        main_details_pick = self._random_pick(main_details_full, num_main_details)
        final_obj["subject"]["main_focus"]["main_details"] = main_details_pick

        # ========== subject->main_focus->main_actions ==========
        main_actions_full = self._get_as_list(merged, ["subject", "main_focus", "main_actions"])
        main_actions_pick = self._random_pick(main_actions_full, num_main_actions)
        final_obj["subject"]["main_focus"]["main_actions"] = main_actions_pick

        # ========== environment->background ==========
        background_full = self._get_as_list(merged, ["environment", "background"])
        background_pick = self._random_pick(background_full, num_background)
        final_obj["environment"]["background"] = background_pick

        # ========== environment->environment_elements ==========
        environment_elements_full = self._get_as_list(merged, ["environment", "environment_elements"])
        environment_elements_pick = self._random_pick(environment_elements_full, num_environment_elements)
        final_obj["environment"]["environment_elements"] = environment_elements_pick

        # ========== environment->actions ==========
        environment_actions_full = self._get_as_list(merged, ["environment", "actions"])
        environment_actions_pick = self._random_pick(environment_actions_full, num_environment_actions)
        final_obj["environment"]["actions"] = environment_actions_pick

        # ========== style ==========
        style_full = self._get_as_list(merged, ["style"])
        style_pick = self._random_pick(style_full, num_style)
        final_obj["style"] = style_pick

        # ========== lighting_and_atmosphere ==========
        lighting_full = self._get_as_list(merged, ["lighting_and_atmosphere"])
        lighting_pick = self._random_pick(lighting_full, num_lighting_atmosphere)
        final_obj["lighting_and_atmosphere"] = lighting_pick

        # ========== others ==========
        others_full = self._get_as_list(merged, ["others"])
        others_pick = self._random_pick(others_full, num_others)
        final_obj["others"] = others_pick

        final_image_json_str = json.dumps(final_obj, indent=2, ensure_ascii=False)
        main_objects_str = json.dumps(main_objects_pick, ensure_ascii=False)
        main_details_str = json.dumps(main_details_pick, ensure_ascii=False)
        main_actions_str = json.dumps(main_actions_pick, ensure_ascii=False)
        background_str = json.dumps(background_pick, ensure_ascii=False)
        environment_elements_str = json.dumps(environment_elements_pick, ensure_ascii=False)
        environment_actions_str = json.dumps(environment_actions_pick, ensure_ascii=False)
        style_str = json.dumps(style_pick, ensure_ascii=False)
        lighting_str = json.dumps(lighting_pick, ensure_ascii=False)
        others_str = json.dumps(others_pick, ensure_ascii=False)

        return (
            final_image_json_str,
            main_objects_str,
            main_details_str,
            main_actions_str,
            background_str,
            environment_elements_str,
            environment_actions_str,
            style_str,
            lighting_str,
            others_str
        )

    def _merge_json_objects(self, objs):
        merged = {}
        if not objs:
            return merged
        all_keys = set()
        for obj in objs:
            all_keys.update(obj.keys())
        for key in all_keys:
            values = [o[key] for o in objs if key in o]
            if not values:
                continue
            if all(isinstance(v, dict) for v in values):
                merged[key] = self._merge_json_objects(values)
            elif all(isinstance(v, list) for v in values):
                combined_list = []
                for v in values:
                    combined_list.extend(v)
                merged[key] = combined_list
            else:
                new_list = []
                for v in values:
                    if isinstance(v, list):
                        new_list.extend(v)
                    else:
                        new_list.append(v)
                merged[key] = new_list
        return merged

    def _get_as_list(self, merged, path):
        cur = merged
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return []
        if isinstance(cur, str):
            return [cur]
        if isinstance(cur, list):
            return cur
        return []

    def _random_pick(self, arr, count):
        if not isinstance(arr, list) or not arr:
            return []
        if count >= len(arr):
            return arr
        return random.sample(arr, count)

class SK_Json_Key_Replace:
    """
    节点名：json_key_replace
    功能：输入目标 JSON 字符串、键路径、键值和模式，对目标 JSON 中同名的键内容进行替换或追加。
    - 用户输入的键值格式为类似 "girl, boy, dogs"，节点会自动将其转换为数组形式：["girl", "boy", "dogs"]。
    - 键路径使用点号分隔来定位层级，例如 "subject.main_focus.main_objects" 或 "environment.background"。
    - 模式可选择 "replace"（替换）或 "append"（追加）。
    - 节点允许处理 4 个替换键，若留空则不做修改。
    """
    CATEGORY = "Sikai_nodes/tools"

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
            data = json.loads(json_str)
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

class SK_Json_Value_Eliminator:
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
    CATEGORY = "Sikai_nodes/tools"

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
            data = json.loads(json_str)
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

class SK_Json_Extract_Key_And_Value:
    """
    节点名：extract_json_key_and_value
    功能：输入 JSON 字符串和最多五个键名，输出指定键名下的值。
         - 键名可以是单个键名（如 main_objects）或嵌套路径（如 subject.main_focus.main_objects）。
         - if_output_key 为 True 时，在输出中保留键名；否则只输出对应的值。
         - flatten 为 True 时，将嵌套列表扁平化后再输出，确保每个子项的抽取概率相等。
         ##2025/03/03##
    """
    CATEGORY = "Sikai_nodes/tools"

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

class SK_Json_Key_Random:
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
    CATEGORY = "Sikai_nodes/tools"

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
            data = json.loads(json_str)
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

class SK_Json_Date_Range_Filter_Recursive:

    CATEGORY = "Sikai_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "start_date": ("STRING", {"default": "2024-01-01", "multiline": False}),
                "end_date": ("STRING", {"default": "2024-12-31", "multiline": False}),
                "target_key": ("STRING", {"default": "date", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_json",)
    FUNCTION = "filter_by_date_range_recursive"

    def filter_by_date_range_recursive(self, json_str, start_date, end_date, target_key):
        # 解析输入的 JSON 字符串
        try:
            data = json.loads(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed: {str(e)}",)
        
        if "image_data" not in data:
            return ("Error: 'image_data' key not found in JSON.",)
        
        image_data = data["image_data"]

        # 转换日期字符串为日期对象
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except Exception as e:
            return (f"Error: Date parsing failed: {str(e)}",)

        # 递归查找目标键对应的值（假设返回第一个找到的值）
        def find_target_value(obj, key):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == key:
                        return v
                    else:
                        res = find_target_value(v, key)
                        if res is not None:
                            return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_target_value(item, key)
                    if res is not None:
                        return res
            return None

        # 筛选符合日期范围的图片数据
        filtered = {}
        for img_id, img_obj in image_data.items():
            date_val = find_target_value(img_obj, target_key)
            if date_val is None:
                continue
            try:
                img_date = datetime.strptime(date_val, "%Y-%m-%d").date()
            except Exception:
                continue
            if start_dt <= img_date <= end_dt:
                filtered[img_id] = img_obj

        output = {"image_data": filtered}
        return (json.dumps(output, indent=2, ensure_ascii=False),)

class SK_Json_Count:
    CATEGORY = "Sikai_nodes/tools"

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
            data = json.loads(json_str)
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

class SK_Word_Frequency_Statistics:
    CATEGORY = "Sikai_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "flatten": ("BOOLEAN", {"default": True}),
                "remove_adjectives": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("frequency_json",)
    FUNCTION = "word_frequency"

    def word_frequency(self, text, flatten, remove_adjectives):
        if flatten:
            # 拆分成单个单词，利用 NLTK 分词
            words = word_tokenize(text.lower())
            if remove_adjectives:
                tagged = pos_tag(words)
                # 过滤掉形容词：JJ, JJR, JJS
                words = [word for word, tag in tagged if tag not in ['JJ', 'JJR', 'JJS']]
        else:
            # 不进一步拆分为单词，而是按照逗号分隔为多个短语
            words = [w.strip().lower() for w in text.split(",") if w.strip()]
        
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        return (json.dumps(freq, indent=2, ensure_ascii=False),)

class SK_Json_Keys_Name:
    CATEGORY = "Sikai_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "multiline": True}),
                "target_object": ("STRING", {"default": "image_data", "multiline": False}),
                "if_output_values": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_keys",)
    FUNCTION = "extract_keys"

    def extract_keys(self, json_str, target_object, if_output_values):
        try:
            data = json.loads(json_str)
        except Exception as e:
            return (f"Error: JSON parsing failed: {str(e)}",)
        
        # 递归搜索目标对象
        target = self.find_target(data, target_object)
        if target is None:
            return (f"Error: Target object '{target_object}' not found.",)
        if not isinstance(target, dict):
            return (f"Error: Target object '{target_object}' is not a dictionary.",)
        
        # 根据 if_output_values 决定输出内容
        if if_output_values:
            result = target  # 输出完整目标对象（键值对）
        else:
            result = list(target.keys())  # 仅输出目标对象中的所有键
        
        return (json.dumps(result, indent=2, ensure_ascii=False),)

    def find_target(self, data, target_object):
        """
        递归搜索 data 中是否存在键名等于 target_object 的元素，
        如果找到，则返回对应的值，否则返回 None。
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if k == target_object:
                    return v
                result = self.find_target(v, target_object)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self.find_target(item, target_object)
                if result is not None:
                    return result
        return None
    
class JsonToString:
    CATEGORY = "Sikai_nodes/tools"

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
        2. 对其中的转义序列做 unicode_escape 解码，
           把 \n 变成换行，把 \" 变成 " 等；
        3. 如果文本最外层有多余的引号，则去除。
        """
        # 步骤 1: 序列化
        try:
            raw = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            raw = repr(data)

        # 步骤 2: 解码转义
        try:
            text = raw.encode('utf-8').decode('unicode_escape')
        except Exception:
            text = raw

        # 步骤 3: 去除最外层引号
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        return (text,)