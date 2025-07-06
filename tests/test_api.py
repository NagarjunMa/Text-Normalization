"""
Integration tests for FastAPI server endpoints
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from api_server_gateway import app, invoke_lambda_function


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def setup_method(self):
        """Setup method for each test"""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Text Normalization API Gateway"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    @patch('api_server_gateway.invoke_lambda_function')
    def test_normalize_single_comment_success(self, mock_invoke):
        """Test successful single comment normalization"""
        # Mock Lambda response
        mock_invoke.return_value = {
            'comment_id': 1,
            'original_text': 'Loan-to-value high. Need bring down to 80.5%.',
            'normalized_text': 'The loan-to-value ratio is high. We need to bring it down to 80.5%.',
            'processing_time': 1.234,
            'lambda_instance_id': 'lambda-123'
        }
        
        payload = {
            'comment_id': 1,
            'text': 'Loan-to-value high. Need bring down to 80.5%.'
        }
        
        response = self.client.post("/normalize", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data['comment_id'] == 1
        assert data['original_text'] == 'Loan-to-value high. Need bring down to 80.5%.'
        assert data['normalized_text'] == 'The loan-to-value ratio is high. We need to bring it down to 80.5%.'
        assert data['processing_time'] > 0
        assert data['lambda_instance_id'] == 'lambda-123'
    
    @patch('api_server_gateway.invoke_lambda_function')
    def test_normalize_single_comment_lambda_error(self, mock_invoke):
        """Test single comment normalization with Lambda error"""
        mock_invoke.side_effect = Exception("Lambda invocation failed")
        
        payload = {
            'comment_id': 1,
            'text': 'Test text'
        }
        
        response = self.client.post("/normalize", json=payload)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"]
    
    def test_normalize_single_comment_invalid_payload(self):
        """Test single comment normalization with invalid payload"""
        payload = {
            'comment_id': 1
            # Missing 'text' field
        }
        
        response = self.client.post("/normalize", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    @patch('api_server_gateway.invoke_lambda_function')
    def test_normalize_batch_comments_success(self, mock_invoke):
        """Test successful batch comment normalization"""
        # Mock Lambda responses
        mock_invoke.side_effect = [
            {
                'comment_id': 1,
                'original_text': 'Comment 1',
                'normalized_text': 'Normalized Comment 1',
                'processing_time': 1.0,
                'lambda_instance_id': 'lambda-1'
            },
            {
                'comment_id': 2,
                'original_text': 'Comment 2',
                'normalized_text': 'Normalized Comment 2',
                'processing_time': 1.5,
                'lambda_instance_id': 'lambda-2'
            }
        ]
        
        payload = {
            'comments': [
                {'comment_id': 1, 'text': 'Comment 1'},
                {'comment_id': 2, 'text': 'Comment 2'}
            ]
        }
        
        response = self.client.post("/normalize-batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 2
        assert data['results'][0]['comment_id'] == 1
        assert data['results'][1]['comment_id'] == 2
    
    @patch('api_server_gateway.invoke_lambda_function')
    def test_normalize_batch_comments_partial_failure(self, mock_invoke):
        """Test batch comment normalization with partial failures"""
        # Mock Lambda responses - one success, one failure
        mock_invoke.side_effect = [
            {
                'comment_id': 1,
                'original_text': 'Comment 1',
                'normalized_text': 'Normalized Comment 1',
                'processing_time': 1.0,
                'lambda_instance_id': 'lambda-1'
            },
            Exception("Lambda error for comment 2")
        ]
        
        payload = {
            'comments': [
                {'comment_id': 1, 'text': 'Comment 1'},
                {'comment_id': 2, 'text': 'Comment 2'}
            ]
        }
        
        response = self.client.post("/normalize-batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 2
        
        # Check successful result
        assert data['results'][0]['comment_id'] == 1
        assert data['results'][0]['normalized_text'] == 'Normalized Comment 1'
        
        # Check failed result
        assert data['results'][1]['comment_id'] == 2
        assert 'Error:' in data['results'][1]['normalized_text']
    
    def test_normalize_batch_comments_empty_list(self):
        """Test batch comment normalization with empty list"""
        payload = {
            'comments': []
        }
        
        response = self.client.post("/normalize-batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 0


class TestLambdaIntegration:
    """Test cases for Lambda function integration"""
    
    @patch('api_server_gateway.lambda_client.invoke')
    def test_invoke_lambda_function_success(self, mock_invoke):
        """Test successful Lambda function invocation"""
        from api_server_gateway import CommentRequest
        
        # Mock Lambda response
        mock_response = Mock()
        mock_response['StatusCode'] = 200
        mock_response['Payload'].read.return_value = json.dumps({
            'comment_id': 1,
            'original_text': 'Test text',
            'normalized_text': 'Normalized text',
            'processing_time': 1.0,
            'lambda_instance_id': 'lambda-123'
        })
        mock_invoke.return_value = mock_response
        
        comment = CommentRequest(comment_id=1, text='Test text')
        result = invoke_lambda_function(comment)
        
        assert result['comment_id'] == 1
        assert result['original_text'] == 'Test text'
        assert result['normalized_text'] == 'Normalized text'
        mock_invoke.assert_called_once()
    
    @patch('api_server_gateway.lambda_client.invoke')
    def test_invoke_lambda_function_api_gateway_response(self, mock_invoke):
        """Test Lambda function invocation with API Gateway response format"""
        from api_server_gateway import CommentRequest
        
        # Mock API Gateway response format
        mock_response = Mock()
        mock_response['StatusCode'] = 200
        mock_response['Payload'].read.return_value = json.dumps({
            'body': json.dumps({
                'comment_id': 1,
                'original_text': 'Test text',
                'normalized_text': 'Normalized text',
                'processing_time': 1.0,
                'lambda_instance_id': 'lambda-123'
            })
        })
        mock_invoke.return_value = mock_response
        
        comment = CommentRequest(comment_id=1, text='Test text')
        result = invoke_lambda_function(comment)
        
        assert result['comment_id'] == 1
        assert result['original_text'] == 'Test text'
        assert result['normalized_text'] == 'Normalized text'
    
    @patch('api_server_gateway.lambda_client.invoke')
    def test_invoke_lambda_function_failure(self, mock_invoke):
        """Test Lambda function invocation failure"""
        from api_server_gateway import CommentRequest
        from fastapi import HTTPException
        
        # Mock Lambda failure
        mock_response = Mock()
        mock_response['StatusCode'] = 500
        mock_response['Payload'].read.return_value = json.dumps({
            'error': 'Lambda execution failed'
        })
        mock_invoke.return_value = mock_response
        
        comment = CommentRequest(comment_id=1, text='Test text')
        
        with pytest.raises(HTTPException) as exc_info:
            invoke_lambda_function(comment)
        
        assert exc_info.value.status_code == 500
        assert "Lambda invocation failed" in str(exc_info.value.detail)


class TestConcurrency:
    """Test cases for concurrent processing"""
    
    @patch('api_server_gateway.invoke_lambda_function')
    def test_concurrent_lambda_invocations(self, mock_invoke):
        """Test concurrent Lambda invocations"""
        from api_server_gateway import invoke_lambda_concurrent
        
        # Mock Lambda responses
        mock_invoke.side_effect = [
            {
                'comment_id': 1,
                'original_text': 'Comment 1',
                'normalized_text': 'Normalized 1',
                'processing_time': 1.0,
                'lambda_instance_id': 'lambda-1'
            },
            {
                'comment_id': 2,
                'original_text': 'Comment 2',
                'normalized_text': 'Normalized 2',
                'processing_time': 1.5,
                'lambda_instance_id': 'lambda-2'
            }
        ]
        
        from api_server_gateway import CommentRequest
        results = [None, None]
        
        # Test concurrent execution
        invoke_lambda_concurrent(CommentRequest(comment_id=1, text='Comment 1'), results, 0)
        invoke_lambda_concurrent(CommentRequest(comment_id=2, text='Comment 2'), results, 1)
        
        assert results[0] is not None
        assert results[1] is not None
        assert results[0]['comment_id'] == 1
        assert results[1]['comment_id'] == 2


if __name__ == "__main__":
    pytest.main([__file__]) 