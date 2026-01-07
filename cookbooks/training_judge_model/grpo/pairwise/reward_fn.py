import torch
import json
import re
from datetime import datetime
import os
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


def extract_preference_response(response_text):
    """
    从模型回复中提取preference偏好
    从<better>标签中提取偏好选择
    """
    # Handle case where response_text might not be a string
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    # 从<better>标签中提取偏好
    preference_pattern = r'<better>(.*?)</better>'
    match = re.search(preference_pattern, response_text, re.DOTALL)
    
    if match:
        preference_content = match.group(1).strip().upper()
        
        # 首先检查是否直接是A或B
        if preference_content == 'A':
            return 'A'
        elif preference_content == 'B':
            return 'B'
        elif preference_content == 'TIE':
            return 'tie'
        
        # 然后检查是否包含特定词汇但不是两者都有
        if 'A' in preference_content and 'B' not in preference_content:
            return 'A'
        elif 'B' in preference_content and 'A' not in preference_content:
            return 'B'
        elif 'TIE' in preference_content or ('A' in preference_content and 'B' in preference_content):
            return 'tie'
    
    # 如果没有找到标签，尝试从文本最后部分提取
    lines = response_text.strip().split('\n')
    for line in reversed(lines[-5:]):  # 检查最后5行
        line = line.strip().upper()
        if line == 'A' or 'RESPONSE A' in line or 'ANSWER A' in line:
            return 'A'
        elif line == 'B' or 'RESPONSE B' in line or 'ANSWER B' in line:
            return 'B'
        elif 'TIE' in line or 'EQUAL' in line:
            return 'tie'
    
    return 'unknown'  # 如果无法提取，返回unknown


def calculate_pairwise_reward(predicted_preference, true_preference, response_id):
    """
    基于preference预测与真实偏好的匹配程度计算奖励
    
    Args:
        predicted_preference: 模型预测的偏好 ('A', 'B', 'tie', 'unknown')
        true_preference: 真实偏好 ('A', 'B', 'tie')
        response_id: 当前response的ID ('A' 或 'B')
    
    Returns:
        float: 奖励分数 (1.0 如果预测正确，0.0 如果预测错误)
    """
    if true_preference is None or predicted_preference == 'unknown':
        return 0.0
    
    # 简化奖励逻辑：预测正确给1分，预测错误给0分
    if predicted_preference == true_preference:
        return 1.0
    else:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    与naive.py兼容的compute_score函数，处理pairwise比较任务
    
    参数：
    - data_source: 数据源类型
    - solution_str: 模型生成的回复
    - ground_truth: 真实标签（包含偏好信息）
    - extra_info: 额外信息
    """
    try:
        # 先过滤掉思考部分（支持Qwen3等模型的thinking模式）
        filtered_solution = filter_thinking_parts(solution_str)
        
        # 从过滤后的solution_str中提取preference
        predicted_preference = extract_preference_response(filtered_solution)
        
        # 处理ground_truth - 应该包含偏好信息
        if isinstance(ground_truth, dict):
            true_preference = ground_truth.get('preference', 'tie')
            response_id = ground_truth.get('response_id', 'A')
            preference_strength = ground_truth.get('preference_strength', 0)
            task_type = ground_truth.get('task_type', 'pairwise')
        else:
            # 回退处理
            if extra_info and isinstance(extra_info, dict):
                # 尝试从extra_info中获取偏好信息
                data_mode = extra_info.get('data_mode', 'pointwise')
                if data_mode == 'pairwise':
                    # 分析原始数据
                    output_data = extra_info.get('output', [])
                    if output_data and len(output_data) >= 2:
                        # 从原始标签中推断偏好
                        label_a = output_data[0].get('answer', {}).get('label', {})
                        label_b = output_data[1].get('answer', {}).get('label', {})
                        
                        pref_a = label_a.get('overall_preference', 0)
                        pref_b = label_b.get('overall_preference', 0)
                        
                        if pref_a > pref_b:
                            true_preference = 'A'
                        elif pref_b > pref_a:
                            true_preference = 'B'
                        else:
                            true_preference = 'tie'
                        
                        # 假设我们在评估第一个response (A)
                        response_id = 'A'
                        preference_strength = abs(pref_a - pref_b)
                        task_type = 'pairwise'
                    else:
                        true_preference = 'tie'
                        response_id = 'A'
                        preference_strength = 0
                        task_type = 'pairwise'
                else:
                    # 不是pairwise任务，返回默认值
                    return {
                        "score": 0.0,
                        "error": "Not a pairwise task",
                        "data_source": data_source
                    }
            else:
                true_preference = 'tie'
                response_id = 'A'
                preference_strength = 0
                task_type = 'pairwise'
        
        # 计算奖励
        reward = calculate_pairwise_reward(predicted_preference, true_preference, response_id)
        
        # 计算准确率
        accuracy = 1.0 if (predicted_preference == true_preference and predicted_preference != 'unknown') else 0.0

        # 返回详细信息
        return {
            "score": reward,
            "predicted_preference": predicted_preference,
            "accuracy": accuracy,
            "true_preference": true_preference,
            "response_id": response_id,
            "preference_strength": preference_strength,
            "task_type": task_type,
            "data_source": data_source
        }
        
    except Exception as e:
        print(f"Error in compute_score: {e}")
        # 返回默认值
        return {
            "score": 0.0,
             "accuracy": 0.0,
            "error": str(e),
            "data_source": data_source
        }


if __name__ == "__main__":
    # 测试用例 - 模拟模型的实际输出
    model_response = '''<think>Let me analyze both responses based on the given principles:

1. Helpfulness: Response A provides detailed step-by-step instructions including washing, peeling, cutting, soaking, and drying. Response B only mentions cutting and frying, missing crucial preparation steps.

2. Accuracy: Response A is factually correct about the soaking process to remove starch. Response B, while not incorrect, lacks important details.

3. Clarity: Response A is clear and well-structured. Response B is clear but overly brief.

4. Completeness: Response A covers all necessary preparation steps. Response B is incomplete, missing several important steps.

5. Relevance: Both responses are relevant, but Response A is more comprehensive in addressing the question.

Response A is significantly better as it provides complete, accurate, and helpful instructions for preparing potatoes for frying.
</think>
<better>A</better>'''
    
    # 测试better标签提取
    extracted_pref = extract_preference_response(model_response)
    print(f"提取的偏好: {extracted_pref}")
    
    # 模拟ground_truth数据
    ground_truth = {
        "preference": "A",
        "preference_strength": 2,
        "response_id": "A",
        "task_type": "pairwise"
    }
    
    # 测试reward计算
    result = compute_score("helpsteer3", model_response, ground_truth)
    print(f"Reward result: {result}")
    
    # 测试不同的预测结果
    test_cases = [
        ("A", "A", "A"),  # 正确预测A更好，当前是A
        ("A", "A", "B"),  # 正确预测A更好，当前是B
        ("B", "A", "A"),  # 错误预测B更好，当前是A
        ("tie", "A", "A"), # 预测tie，真实A更好，当前是A
    ]
    
    print("\n=== 测试不同预测结果 ===")
    for pred, true, resp_id in test_cases:
        test_gt = {
            "preference": true,
            "preference_strength": 1,
            "response_id": resp_id,
            "task_type": "pairwise"
        }
        reward = calculate_pairwise_reward(pred, true, resp_id)
        print(f"预测: {pred}, 真实: {true}, Response ID: {resp_id} -> 奖励: {reward:.1f}") 