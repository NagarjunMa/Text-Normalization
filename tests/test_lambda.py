"""
Unit tests for Lambda function text normalization
"""
import pytest
import json
import boto3
from unittest.mock import Mock, patch, MagicMock
from lambda_function import BedrockTextNormalizer, lambda_handler


class TestBedrockTextNormalizer:
    """Test cases for BedrockTextNormalizer class"""
    
    def setup_method(self):
        """Setup method for each test"""
        self.normalizer = BedrockTextNormalizer()
    
    def test_create_prompt(self):
        """Test prompt creation"""
        text = "Loan-to-value high. Need bring down to 80.5%."
        prompt = self.normalizer.create_prompt(text)
        
        assert "normalize" in prompt.lower()
        assert text in prompt
        assert "professional" in prompt.lower()
        assert "insurance" in prompt.lower()
    
    def test_extract_numbers_before_llm(self):
        """Test number extraction and preservation"""
        text = "Loan-to-value high. Need bring down to 80.5%. Risk too big."
        modified_text, placeholder_map = self.normalizer.extract_numbers_before_llm(text)
        
        # Check that numbers are replaced with placeholders
        assert "__NUMBER_" in modified_text
        assert "80.5%" in placeholder_map.values()
        
        # Check that original text is preserved in placeholders
        assert len(placeholder_map) > 0
    
    def test_restore_numbers_after_llm(self):
        """Test number restoration after LLM processing"""
        placeholder_map = {"__NUMBER_0__": "80.5%", "__NUMBER_1__": "$500"}
        text_with_placeholders = "The ratio is __NUMBER_0__ and cost is __NUMBER_1__"
        
        restored_text = self.normalizer.restore_numbers_after_llm(
            text_with_placeholders, placeholder_map
        )
        
        assert "80.5%" in restored_text
        assert "$500" in restored_text
        assert "__NUMBER_" not in restored_text
    
    @patch('lambda_function.bedrock_runtime.invoke_model')
    def test_normalize_with_bedrock_success(self, mock_invoke):
        """Test successful Bedrock normalization"""
        # Mock successful Bedrock response
        mock_response = Mock()
        mock_response['body'].read.return_value = json.dumps({
            'output': {
                'message': {
                    'content': [{'text': 'The loan-to-value ratio is high. We need to bring it down to 80.5%.'}]
                }
            }
        })
        mock_invoke.return_value = mock_response
        
        text = "Loan-to-value high. Need bring down to 80.5%."
        result = self.normalizer.normalize_with_bedrock(text)
        
        assert "loan-to-value" in result.lower()
        assert "80.5%" in result
        mock_invoke.assert_called_once()
    
    @patch('lambda_function.bedrock_runtime.invoke_model')
    def test_normalize_with_bedrock_error(self, mock_invoke):
        """Test Bedrock API error handling"""
        mock_invoke.side_effect = Exception("Bedrock API error")
        
        text = "Test text"
        with pytest.raises(Exception, match="Bedrock API error"):
            self.normalizer.normalize_with_bedrock(text)


class TestLambdaHandler:
    """Test cases for Lambda handler function"""
    
    def test_lambda_handler_success(self):
        """Test successful Lambda handler execution"""
        event = {
            'comment_id': 1,
            'text': 'Loan-to-value high. Need bring down to 80.5%.'
        }
        
        with patch('lambda_function.BedrockTextNormalizer') as mock_normalizer_class:
            mock_normalizer = Mock()
            mock_normalizer.normalize_with_bedrock.return_value = "Normalized text"
            mock_normalizer_class.return_value = mock_normalizer
            
            result = lambda_handler(event, {})
            
            assert result['statusCode'] == 200
            body = json.loads(result['body'])
            assert body['comment_id'] == 1
            assert body['normalized_text'] == "Normalized text"
            assert 'processing_time' in body
    
    def test_lambda_handler_missing_text(self):
        """Test Lambda handler with missing text"""
        event = {'comment_id': 1}
        
        result = lambda_handler(event, {})
        
        assert result['statusCode'] == 400
        body = json.loads(result['body'])
        assert 'error' in body
    
    def test_lambda_handler_api_gateway_event(self):
        """Test Lambda handler with API Gateway event format"""
        event = {
            'body': json.dumps({
                'comment_id': 1,
                'text': 'Test text'
            })
        }
        
        with patch('lambda_function.BedrockTextNormalizer') as mock_normalizer_class:
            mock_normalizer = Mock()
            mock_normalizer.normalize_with_bedrock.return_value = "Normalized text"
            mock_normalizer_class.return_value = mock_normalizer
            
            result = lambda_handler(event, {})
            
            assert result['statusCode'] == 200
            body = json.loads(result['body'])
            assert body['comment_id'] == 1
    
    def test_lambda_handler_exception(self):
        """Test Lambda handler exception handling"""
        event = {
            'comment_id': 1,
            'text': 'Test text'
        }
        
        with patch('lambda_function.BedrockTextNormalizer') as mock_normalizer_class:
            mock_normalizer = Mock()
            mock_normalizer.normalize_with_bedrock.side_effect = Exception("Test error")
            mock_normalizer_class.return_value = mock_normalizer
            
            result = lambda_handler(event, {})
            
            assert result['statusCode'] == 500
            body = json.loads(result['body'])
            assert 'error' in body


class TestNumberPreservation:
    """Test cases for number preservation functionality"""
    
    def setup_method(self):
        """Setup method for each test"""
        self.normalizer = BedrockTextNormalizer()
    
    def test_percentage_preservation(self):
        """Test percentage number preservation"""
        text = "Need to reduce premium by 15.5%"
        modified_text, placeholder_map = self.normalizer.extract_numbers_before_llm(text)
        
        assert "15.5%" in placeholder_map.values()
        assert "__NUMBER_" in modified_text
    
    def test_currency_preservation(self):
        """Test currency amount preservation"""
        text = "Claim amount is $2,500.75"
        modified_text, placeholder_map = self.normalizer.extract_numbers_before_llm(text)
        
        assert "$2,500.75" in placeholder_map.values()
        assert "__NUMBER_" in modified_text
    
    def test_range_preservation(self):
        """Test number range preservation"""
        text = "Risk score between 7.5 ~ 8.2"
        modified_text, placeholder_map = self.normalizer.extract_numbers_before_llm(text)
        
        assert "7.5 ~ 8.2" in placeholder_map.values()
        assert "__NUMBER_" in modified_text
    
    def test_multiple_numbers(self):
        """Test multiple number preservation"""
        text = "Premium $500, deductible 15%, coverage 80.5%"
        modified_text, placeholder_map = self.normalizer.extract_numbers_before_llm(text)
        
        assert len(placeholder_map) >= 3
        assert "$500" in placeholder_map.values()
        assert "15%" in placeholder_map.values()
        assert "80.5%" in placeholder_map.values()


if __name__ == "__main__":
    pytest.main([__file__]) 