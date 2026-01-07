# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from typing import List, Union

import datasets
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

# 注意：已移除 pydantic 模板类以避免 Ray pickle 序列化问题


class BaseChatRLDataset(Dataset):
    """聊天强化学习数据集基类"""

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,  # 保持向后兼容性，但不使用
        max_samples: int = -1,  # 添加 max_samples 参数
    ):
        # 初始化基本属性
        self.data_files = self._normalize_data_files(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        
        # 加载配置设置
        self._load_config()
        
        # 加载和处理数据
        self._load_dataset()

    def _normalize_data_files(self, data_files):
        """将数据文件转换为列表格式"""
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        return copy.deepcopy(data_files)

    def _load_config(self):
        """加载配置参数"""
        self.cache_dir = os.path.expanduser(self.config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.truncation = self.config.get("truncation", "error")
        self.filter_overlong_prompts = self.config.get("filter_overlong_prompts", True)
        self.num_workers = min(
            self.config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)),
            os.cpu_count()
        )
        self.serialize_dataset = False

    def _download_files(self):
        """下载文件到本地缓存"""
        from verl.utils.fs import copy_to_local
        
        for i, file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=file, cache_dir=self.cache_dir)

    def _load_dataset(self):
        """加载和处理数据集"""
        self._download_files()
        
        # 加载parquet文件
        dataframes = []
        for file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=file)["train"]
            dataframes.append(df)
        
        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"数据集长度: {total}")
        
        # 处理 max_samples 参数
        if self.max_samples > 0 and self.max_samples < total:
            import numpy as np
            indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"选择了 {self.max_samples} 个样本（共 {total} 个）")
        
        # 过滤过长的提示
        if self.filter_overlong_prompts:
            self._filter_long_prompts()

    def _filter_long_prompts(self):
        """过滤掉过长的提示"""
        # 提取 tokenizer 和参数到局部变量，避免 pickle 序列化问题
        tokenizer = self.tokenizer
        max_length = self.max_prompt_length
        prompt_key = self.prompt_key
        
        def is_prompt_valid(doc):
            try:
                # 内联提取 prompt 逻辑，避免调用 self 方法
                prompt = ""
                if "input" in doc and doc["input"]:
                    for msg in doc["input"]:
                        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                            prompt = msg["content"]
                            break
                
                if not prompt:
                    # 回退到其他字段
                    prompt = doc.get(prompt_key, "")
                    if isinstance(prompt, list) and prompt:
                        prompt = prompt[0].get("content", "") if isinstance(prompt[0], dict) else str(prompt[0])
                
                if not prompt:
                    return True  # 如果无法提取 prompt，保留该样本
                
                return len(tokenizer.encode(prompt)) <= max_length
            except Exception as e:
                print(f"过滤时出错: {e}")
                return True  # 出错时保留该样本
        
        original_len = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            is_prompt_valid,
            num_proc=1,  # 使用单进程避免序列化问题
            desc=f"过滤长度超过 {max_length} tokens的提示",
        )
        print(f"过滤后数据集长度: {len(self.dataframe)} (原始: {original_len})")

    def _extract_prompt(self, example):
        """从样本中提取提示"""
        # 首先尝试新的数据结构
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if msg.get("role") == "user" and msg.get("content"):
                    return msg["content"]
        
        # 回退到旧的数据结构
        prompt = example.get(self.prompt_key)
        if prompt is None:
            prompt = example.get("x", [])
            if prompt:
                return prompt[-1].get("content", "")
        
        if isinstance(prompt, str):
            return prompt[:self.max_prompt_length]
        elif isinstance(prompt, list) and prompt:
            return prompt[0].get("content", "") if isinstance(prompt[0], dict) else str(prompt[0])
        
        return ""

    def _build_messages(self, example: dict) -> List[dict]:
        """从样本构建聊天消息 - 子类需要重写"""
        raise NotImplementedError("Subclasses must implement _build_messages")

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """格式化模板 - 子类需要重写"""
        raise NotImplementedError("Subclasses must implement _format_template")

    def _extract_ground_truth(self, row_dict):
        """提取真实标签 - 子类需要重写"""
        raise NotImplementedError("Subclasses must implement _extract_ground_truth")

    def __getitem__(self, item):
        """获取数据集中的一个项目"""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)
        
        # 格式化提示
        raw_prompt_messages = self._format_template(messages, row_dict)

        # 尝试使用 enable_thinking 参数，如果不支持则回退
        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                raw_prompt_messages, 
                add_generation_prompt=True, 
                tokenize=False, 
                enable_thinking=True
            )
        except TypeError:
            # 如果 tokenizer 不支持 enable_thinking 参数，则不使用
            raw_prompt = self.tokenizer.apply_chat_template(
                raw_prompt_messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        
        # 分词
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        
        # 后处理
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        # 计算位置ID
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # 准备原始提示ID
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"提示长度 {len(raw_prompt_ids)} 超过 {self.max_prompt_length}")
        
        # 构建结果
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": row_dict.get("source", "helpsteer2"),
        }
        
        if self.return_raw_chat:
            result["raw_prompt"] = messages
            
        return result

    def __len__(self):
        return len(self.dataframe)

    def resume_dataset_state(self):
        """恢复数据集状态用于检查点"""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self.data_files = copy.deepcopy(self.original_data_files)
            self._load_dataset()
        else:
            print("使用旧的数据加载器检查点文件，建议从头开始训练")

    def __getstate__(self):
        """获取用于序列化的状态"""
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()


class PairwiseChatRLDataset(BaseChatRLDataset):
    """Pairwise聊天强化学习数据集"""
    
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        # Pairwise相关配置
        self.pairwise_response_index = self.config.get("pairwise_response_index", 0)  # 选择哪个response进行训练
        print(f"使用 Pairwise 模式，选择 response index: {self.pairwise_response_index}")

    def _build_messages(self, example: dict) -> List[dict]:
        """从样本构建聊天消息 - Pairwise模式"""
        messages = []
        
        # 从input字段提取用户消息
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if msg.get("role") == "user" and msg.get("content"):
                    messages.append({"role": "user", "content": msg["content"]})
        
        # Pairwise模式：选择指定的response
        if "output" in example and example["output"]:
            if self.pairwise_response_index < len(example["output"]):
                output_item = example["output"][self.pairwise_response_index]
                answer = output_item.get("answer", {})
                if isinstance(answer, dict) and answer.get("role") == "assistant":
                    content = answer.get("content", "")
                    if content:
                        messages.append({"role": "assistant", "content": content})
        
        # 回退到原始结构
        if len(messages) <= 1:
            prompt = self._extract_prompt(example)
            if prompt:
                messages = [{"role": "user", "content": prompt}]
        
        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """格式化pairwise模板"""
        task_desc = """You are a professional expert in response comparison.
You will be provided with a query and two different responses (A and B) to that query.
Your task is to determine which response is better by comparing their quality across multiple dimensions.
Please consider the following principles in your evaluation and then indicate your preference."""

        principles = [
            "Helpfulness: How well does the response address the user's needs",
            "Accuracy: Factual correctness and reliability of information",
            "Safety: Avoiding harmful or inappropriate content",
        ]
        
        # 提取问题
        query = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
        
        # 获取两个回答
        response_a = ""
        response_b = ""
        
        if "output" in example and len(example["output"]) >= 2:
            response_a = example["output"][0].get("answer", {}).get("content", "")
            response_b = example["output"][1].get("answer", {}).get("content", "")
        
        # 直接使用字符串格式化，避免使用 PairwiseTrainTemplate 类（防止 pickle 序列化问题）
        principles_str = ""
        for i, principle in enumerate(principles):
            principles_str += f"{i + 1}. {principle}\n"
        
        prompt = f"""# Task Description
{task_desc}
# Principles
{principles_str}
# Examples

# Query
{query}
# Response A
{response_a}
# Response B
{response_b}
# Output Format
<think>Analysis process based on principles</think><better>A or B</better>
"""
        return [{"role": "user", "content": prompt}]

    def _extract_ground_truth(self, row_dict):
        """提取pairwise真实标签"""
        try:
            output_data = row_dict.get("output", [])
            if output_data and len(output_data) >= 2:
                # 获取选中response的标签
                selected_answer = output_data[self.pairwise_response_index].get("answer", {})
                if isinstance(selected_answer, dict):
                    label_data = selected_answer.get("label", {})
                    if isinstance(label_data, dict):
                        # 对于pairwise，返回偏好信息
                        preference = label_data.get("preference", "")
                        strength = label_data.get("preference_strength", 0)
                        response_id = label_data.get("response_id", "")
                        
                        return {
                            "preference": preference,
                            "preference_strength": strength,
                            "response_id": response_id,
                            "task_type": "pairwise"
                        }
            
            return ""
        except:
            return ""


class PointwiseChatRLDataset(BaseChatRLDataset):
    """Pointwise聊天强化学习数据集 - 用于单个回答的质量评分"""
    
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        print(f"使用 Pointwise 模式")

    def _build_messages(self, example: dict) -> List[dict]:
        """从样本构建聊天消息 - Pointwise模式"""
        messages = []
        
        # 从input字段提取用户消息
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    messages.append({"role": "user", "content": msg["content"]})
        
        # Pointwise模式：获取第一个response
        if "output" in example and example["output"]:
            output_item = example["output"][0] if isinstance(example["output"], list) else example["output"]
            answer = output_item.get("answer", {}) if isinstance(output_item, dict) else {}
            if isinstance(answer, dict) and answer.get("role") == "assistant":
                content = answer.get("content", "")
                if content:
                    messages.append({"role": "assistant", "content": content})
        
        # 回退到原始结构
        if len(messages) <= 1:
            prompt = self._extract_prompt(example)
            if prompt:
                messages = [{"role": "user", "content": prompt}]
        
        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """格式化pointwise模板"""
        task_desc = """You are a professional expert in response quality evaluation.
You will be provided with a query and a response to that query.
Your task is to evaluate the quality of the response and assign a helpfulness score from 0 to 4.
Please consider the following principles in your evaluation."""

        principles = [
            "Helpfulness: How well does the response address the user's needs (0=not helpful, 4=extremely helpful)",
            "Accuracy: Factual correctness and reliability of information",
            "Clarity: How clearly and understandably the response is written",
            "Completeness: Whether the response fully addresses all aspects of the question",
            "Relevance: How directly related the response is to the question",
            "Safety: Avoiding harmful or inappropriate content",
        ]
        
        # 提取问题
        query = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
        
        # 获取回答
        response = ""
        if "output" in example and example["output"]:
            output_item = example["output"][0] if isinstance(example["output"], list) else example["output"]
            if isinstance(output_item, dict):
                response = output_item.get("answer", {}).get("content", "")
        
        # 直接使用字符串格式化
        principles_str = ""
        for i, principle in enumerate(principles):
            principles_str += f"{i + 1}. {principle}\n"
        
        prompt = f"""# Task Description
{task_desc}
# Principles
{principles_str}
# Query
{query}
# Response
{response}
# Output Format
<think>Analysis process based on principles</think><score>helpfulness score from 0 to 4</score>
"""
        return [{"role": "user", "content": prompt}]

    def _extract_ground_truth(self, row_dict):
        """提取pointwise真实标签"""
        try:
            output_data = row_dict.get("output", [])
            if output_data:
                output_item = output_data[0] if isinstance(output_data, list) else output_data
                if isinstance(output_item, dict):
                    answer = output_item.get("answer", {})
                    if isinstance(answer, dict):
                        label_data = answer.get("label", {})
                        if isinstance(label_data, dict):
                            # 对于pointwise，返回评分信息
                            helpfulness = label_data.get("helpfulness", 0)
                            return {
                                "helpfulness": helpfulness,
                                "task_type": "pointwise"
                            }
            
            return {"helpfulness": 0, "task_type": "pointwise"}
        except:
            return {"helpfulness": 0, "task_type": "pointwise"}


# 向后兼容的别名
