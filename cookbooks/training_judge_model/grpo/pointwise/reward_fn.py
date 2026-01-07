import torch
import json
from datetime import datetime
import os
import re
from collections import defaultdict


def filter_thinking_parts(text):
    """
    过滤文本中的思考部分（用于Qwen3等支持thinking模式的模型）
    
    支持的思考标记格式：
    - <think>...</think>
    """
    if not isinstance(text, str):
        return text
    
    # 定义思考部分的正则表达式模式
    thinking_patterns = [
        r'<think>.*?</think>'
    ]
    
    # 依次应用所有模式进行过滤
    filtered_text = text
    for pattern in thinking_patterns:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余的空白字符
    filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)  # 合并多个换行
    filtered_text = filtered_text.strip()
    
    return filtered_text


def extract_helpfulness_score(response_text):
    """
    从模型回复中提取helpfulness评分
    从<score>标签中提取分数
    """
    # Handle case where response_text might not be a string
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    # 从<score>标签中提取分数
    score_pattern = r'<score>(.*?)</score>'
    match = re.search(score_pattern, response_text, re.DOTALL)
    
    if match:
        score_content = match.group(1).strip()
        # 提取其中的数字
        numbers = re.findall(r'\d+', score_content)
        if numbers:
            try:
                score = int(numbers[0])  # 取第一个数字作为分数
                if 0 <= score <= 4:  # 假设分数范围是0-4
                    return score
            except:
                pass
    
    return 0  # 如果无法提取，默认为0

def calculate_helpfulness_reward(predicted_score, true_score):
    """
    基于helpfulness预测分数与真实分数的差异计算奖励
    差异越小，奖励越高
    
    对于二分类场景 (true_score为0或1):
    - 预测正确（完全匹配）→ 奖励1.0
    - 预测错误 → 奖励0.0
    """
    if true_score is None:
        return 0.0
    
    # 计算差异
    diff = abs(predicted_score - true_score)
    
    # 对于二分类场景（0或1），采用精确匹配
    if true_score in [0, 1]:
        return 1.0 if diff == 0 else 0.0
    
    # 对于多分类场景（0-4），采用差异计算
    # 将差异转换为奖励分数 (差异越小奖励越高)
    max_possible_diff = 4
    normalized_diff = min(diff / max_possible_diff, 1.0)
    
    # 奖励 = 1 - 标准化差异
    reward = 1.0 - normalized_diff
    
    return reward

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    与naive.py兼容的compute_score函数
    参数：
    - data_source: 数据源类型
    - solution_str: 模型生成的回复
    - ground_truth: 真实标签（从reward_model字段获取）
    - extra_info: 额外信息
    """
    try:
        # 先过滤掉思考部分（支持Qwen3等模型的thinking模式）
        filtered_solution = filter_thinking_parts(solution_str)
        
        # 从过滤后的solution_str中提取helpfulness分数
        predicted_helpfulness = extract_helpfulness_score(filtered_solution)
        
        # 处理ground_truth - 可能是数字或者字典
        if isinstance(ground_truth, dict):
            true_helpfulness = ground_truth.get('helpfulness', 0)
        elif isinstance(ground_truth, (int, float)):
            true_helpfulness = int(ground_truth)
        elif isinstance(ground_truth, str) and ground_truth.isdigit():
            true_helpfulness = int(ground_truth)
        else:
            # 如果ground_truth不可用，尝试从extra_info中获取
            if extra_info and isinstance(extra_info, dict):
                output_data = extra_info.get('output', [])
                if output_data and len(output_data) > 0:
                    label_data = output_data[0].get('label', {})
                    true_helpfulness = label_data.get('helpfulness', 0)
                else:
                    true_helpfulness = 0
            else:
                true_helpfulness = 0
        
        # 计算奖励
        reward = calculate_helpfulness_reward(predicted_helpfulness, true_helpfulness)
        
        # 返回详细信息
        return {
            "score": reward,
            "predicted_helpfulness": predicted_helpfulness,
            "true_helpfulness": true_helpfulness,
            "data_source": data_source
        }
        
    except Exception as e:
        print(f"Error in compute_score: {e}")
        # 返回默认值
        return {
            "score": 0.0,
            "error": str(e),
            "data_source": data_source
        }

if __name__ == "__main__":
    # 测试用例
    test_response = '''<think>Let me analyze this answer step by step:
1. First, I'll check if the answer is well-structured...
4. Finally, I'll look at the overall helpfulness...
</think>
<score>2</score>'''
    
    ground_truth = {"helpfulness": 3, "task_type": "pointwise"}
    
    # 测试 compute_score 函数
    result = compute_score(
        data_source="test",
        solution_str=test_response,
        ground_truth=ground_truth
    )
    
    print(f"Test Result:")
    print(f"  Predicted Score: {result.get('predicted_helpfulness')}")
    print(f"  True Score: {result.get('true_helpfulness')}")
    print(f"  Reward: {result.get('score')}")

