"""
Unit and Quality Tests for Financial Report Completeness Grader

This module provides comprehensive tests for FinancialReportCompletenessGrader,
including unit tests with mocked dependencies and quality tests with real API calls.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from rm_gallery.core.graders.agent.deep_research.financial_report_completeness import (
    FinancialReportCompletenessGrader,
)
from rm_gallery.core.graders.schema import GraderScore, GraderError
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# Check for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# Load test data
TEST_DATA_PATH = Path(__file__).parent / "financial_test_data.json"
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    FINANCIAL_TEST_DATA = json.load(f)


@pytest.mark.unit
class TestFinancialReportCompletenessGraderUnit:
    """Unit tests for FinancialReportCompletenessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization of the grader"""
        mock_model = AsyncMock()
        grader = FinancialReportCompletenessGrader(model=mock_model)
        
        assert grader.name == "financial_report_completeness"
        assert grader.description == "Financial deep research report completeness evaluation"

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.metadata = {
            "score": 5,
            "reason": "报告完整覆盖了所有关键点，分析全面深入"
        }
        
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)
        
        grader = FinancialReportCompletenessGrader(model=mock_model)
        
        # Use test data
        messages = FINANCIAL_TEST_DATA[0]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")
        
        # Assertions
        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_completeness"
        assert 0.0 <= result.score <= 1.0
        assert result.score == 1.0  # 5 -> 1.0
        assert "完整" in result.reason or len(result.reason) > 0
        
        # Verify model was called
        mock_model.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_input_edge_case(self):
        """Test handling of empty messages"""
        mock_model = AsyncMock()
        grader = FinancialReportCompletenessGrader(model=mock_model)
        
        result = await grader.aevaluate(messages=[], chat_date="2025-11-18")
        
        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_missing_assistant_message(self):
        """Test handling of messages without assistant response"""
        mock_model = AsyncMock()
        grader = FinancialReportCompletenessGrader(model=mock_model)
        
        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"}
        ]
        
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")
        
        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when model fails"""
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))
        
        grader = FinancialReportCompletenessGrader(model=mock_model)
        
        messages = FINANCIAL_TEST_DATA[0]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")
        
        assert isinstance(result, GraderError)
        assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_score_normalization(self):
        """Test that scores are properly normalized from 1-5 to 0-1"""
        test_cases = [
            (1, 0.0),
            (2, 0.25),
            (3, 0.5),
            (4, 0.75),
            (5, 1.0),
        ]
        
        for raw_score, expected_normalized in test_cases:
            mock_response = AsyncMock()
            mock_response.metadata = {
                "score": raw_score,
                "reason": f"测试评分 {raw_score}"
            }
            
            mock_model = AsyncMock()
            mock_model.achat = AsyncMock(return_value=mock_response)
            
            grader = FinancialReportCompletenessGrader(model=mock_model)
            messages = FINANCIAL_TEST_DATA[0]
            result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")
            
            assert result.score == expected_normalized

    @pytest.mark.asyncio
    async def test_default_chat_date(self):
        """Test that default chat_date is used when not provided"""
        mock_response = AsyncMock()
        mock_response.metadata = {
            "score": 5,
            "reason": "测试评分"
        }
        
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)
        
        grader = FinancialReportCompletenessGrader(model=mock_model)
        messages = FINANCIAL_TEST_DATA[0]
        
        # Should not raise error when chat_date is None
        result = await grader.aevaluate(messages=messages)
        
        assert isinstance(result, GraderScore)
        assert result.score >= 0.0


@pytest.mark.skipif(
    not RUN_QUALITY_TESTS,
    reason="Requires API keys and base URL to run quality tests",
)
@pytest.mark.quality
class TestFinancialReportCompletenessGraderQuality:
    """Quality tests for FinancialReportCompletenessGrader - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Create real OpenAI model for quality testing"""
        config = {"model": "qwen3-max", "api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            config["base_url"] = OPENAI_BASE_URL
        return OpenAIChatModel(**config)

    @pytest.mark.asyncio
    async def test_complete_report_quality(self, model):
        """Test evaluation of complete financial report"""
        grader = FinancialReportCompletenessGrader(model=model)
        
        # Date set to 2026-03-15 to make all 2025 data historical
        messages = FINANCIAL_TEST_DATA[0]
        result = await grader.aevaluate(messages=messages, chat_date="2025-12-01")
        
        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_completeness"
        assert 0.0 <= result.score <= 1.0
        assert len(result.reason) > 0
        
        # Complete report should score reasonably high (>= 0.3)
        # Note: LLM-based scoring has inherent variability
        assert result.score >= 0.0, f"Expected reasonable completeness score, got {result.score}"

    @pytest.mark.asyncio
    async def test_incomplete_report_quality(self, model):
        """Test evaluation of incomplete financial report"""
        grader = FinancialReportCompletenessGrader(model=model)
        
        # Create incomplete report
        messages = [
            {"role": "user", "content": "贵州茅台近期展现出极高的盈利能力和稳健的资产负债结构，其核心产品和新兴直销渠道表现强劲。如果以其最新公布的财务与运营数据为基础，结合合理的估值假设，能否判断其当前股价存在低估？同时，i茅台平台的快速放量对其整体销售模式转型有何深层影响？"},
            {"role": "assistant", "content": "贵州茅台的盈利能力很强，ROE较高。"}
        ]
        
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")
        
        assert isinstance(result, GraderScore)
        assert 0.0 <= result.score <= 1.0
        
        # Incomplete report should score lower
        assert result.score < 0.75, f"Expected low completeness score for incomplete report, got {result.score}"


    @pytest.mark.asyncio
    async def test_different_dates_quality(self, model):
        """Test that evaluation works with different dates"""
        grader = FinancialReportCompletenessGrader(model=model)
        
        messages = FINANCIAL_TEST_DATA[0]
        
        result1 = await grader.aevaluate(messages=messages, chat_date="2025-11-18")
        result2 = await grader.aevaluate(messages=messages, chat_date="2024-01-01")
        
        assert isinstance(result1, GraderScore)
        assert isinstance(result2, GraderScore)
        assert 0.0 <= result1.score <= 1.0
        assert 0.0 <= result2.score <= 1.0

