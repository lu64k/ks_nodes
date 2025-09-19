import torch
import os
import time
import random

class KSLoadText:
    CATEGORY = "Sikai_nodes/tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "./input.txt", "multiline": False}),  # 文件路径
                "use_seed": ("BOOLEAN", {"default": True}),  # 是否使用随机种子
                "seed": ("INT", {"default": 42}),  # 随机种子值
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_content",)
    FUNCTION = "load_text"

    def load_text(self, file_path, use_seed, seed):
        try:
            # 每次调用节点时强制触发读取逻辑，使用当前时间作为随机数种子
            random.seed(time.time())

            # 检查文件夹是否存在，不存在则创建
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                status_message = f"Directory '{directory}' created."

            # 检查文件是否存在，不存在则创建一个空文件
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")  # 创建空文件
                status_message = f"File '{file_path}' created as it did not exist."

            # 如果选择了使用随机种子，设置种子值
            if use_seed:
                random.seed(seed)
                status_message = f"Seed {seed} applied."

            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # 返回文件内容
            return (file_content,)

        except Exception as e:
            return (f"Error: {str(e)}",)

class KS_Save_Text:
    CATEGORY = "Sikai_nodes/tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "./output.txt", "multiline": False}),  # 文件路径
                "text_content": ("STRING", {"default": "Default content", "multiline": True}),  # 要保存的文本内容
                "save_mode": (["overwrite", "append", "new only"],),  # 保存模式：overwrite, append, new only
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_text"

    def save_text(self, file_path, text_content, save_mode):
        status_message = ""

        try:
            # Check the save mode and handle accordingly
            if save_mode == "overwrite":
                # 1. Overwrite mode - 覆盖现有文件或新建
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                status_message = f"File '{file_path}' was overwritten successfully."

            elif save_mode == "append":
                # 2. Append mode - 追加文本内容到现有文件
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(text_content)
                status_message = f"Text was appended to '{file_path}'."

            elif save_mode == "new only":
                # 3. New only mode - 如果文件已存在，则不进行写入
                if not os.path.exists(file_path):
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    status_message = f"New file '{file_path}' was created."
                else:
                    status_message = f"File '{file_path}' already exists. No changes made."

        except Exception as e:
            status_message = f"Error: {str(e)}"
        return (status_message,)

class KS_Random_File_Name:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": '', "multiline": False}),  # 输入文件夹路径
                "seed": ("INT", {"default": 0}),  # 全局 seed
                "fix_random": ("BOOLEAN", {"default": True}),  # 控制是否固定随机性 (fix/random)
                "index": ("INT", {"default": 0})  # 新增按索引顺序读取的选项
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_name_without_extension", "file_name_with_extension", "full_file_path")
    FUNCTION = "get_file_name"

    CATEGORY = "Sikai_nodes/tools"

    def get_file_name(self, folder_path, seed, fix_random, index):
        """
        根据设定的模式从文件夹中读取文件：
        - 随机读取模式：根据 seed 和 fix_random 控制随机性。
        - 索引顺序模式：按传入的 index 获取文件。
        """

        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            return ("Error: Folder not found", "", "")

        # 获取文件夹中的所有文件并排序（保持顺序一致性）
        files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        # 如果文件夹为空，返回错误信息
        if not files:
            return ("Error: No files in folder", "", "")

        if fix_random:
            # 如果 fix_random 为 True，使用随机模式
            random.seed(seed)
            random_file = random.choice(files)
        else:
            # 如果 fix_random 为 False，按索引顺序读取文件
            # 计算循环索引，确保不会越界
            valid_index = index % len(files)
            random_file = files[valid_index]

        # 获取文件名和扩展名
        file_name_without_extension = os.path.splitext(random_file)[0]
        file_name_with_extension = random_file

        # 获取完整路径
        full_file_path = os.path.join(folder_path, random_file)

        # 返回去除后缀的文件名、带后缀的完整文件名和完整文件路径
        return (file_name_without_extension, file_name_with_extension, full_file_path)

class KS_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            },
            "optional": {
                "text_b": ("STRING", {"default": '', "multiline": True}),
                "text_c": ("STRING", {"default": '', "multiline": True}),
                "text_d": ("STRING", {"default": '', "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_text", "output_text_b", "output_text_c", "output_text_d")
    FUNCTION = "text_string"

    CATEGORY = "Sikai_nodes/tools"

    def text_string(self, text, text_b='', text_c='', text_d=''):
        """
        处理输入的四组文本，并分别返回。
        如果没有提供 text_b, text_c, text_d，使用默认空字符串。
        """
        # 返回四个字符串
        return (text, text_b, text_c, text_d)

class KS_get_time_int:
    CATEGORY = "Sikai_nodes/tools"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max": ("INT", {"default": 1000000000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("time", "time_random")
    FUNCTION = "get_time"

    def IS_CHANGED(self, *args, **kwargs):
        # 每次调用此方法时生成一个随机数，强制节点重新计算
        return torch.rand(1).item()

    def get_time(self, min, max):
        # 强制从系统随机数生成器中取一个内部随机种子，并用它来设置内置随机数生成器的种子
        internal_seed = torch.randint(min, max, (1,)).item()
        random.seed(internal_seed)
        # 获取当前时间戳（秒）
        current_time = int(time.time())
        # 生成指定范围内的随机整数
        time_random = random.randint(min, max)
        return (current_time, time_random)