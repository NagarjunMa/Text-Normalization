import boto3
import json
import time
import asyncio
import threading
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import logging
from config.settings import get_settings

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Normalization API Gateway", version="1.0.0")

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize AWS Lambda client
lambda_client = boto3.client('lambda', region_name=settings.aws_region)

class CommentRequest(BaseModel):
    comment_id: int
    text: str
    
    class Config:
        # Add validation
        str_min_length = 1
        str_max_length = 10000  # Reasonable limit for text input
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v.strip()) > 10000:
            raise ValueError('Text too long (max 10000 characters)')
        return v.strip()
    
    @validator('comment_id')
    def validate_comment_id(cls, v):
        if v <= 0:
            raise ValueError('Comment ID must be positive')
        return v

class CommentResponse(BaseModel):
    comment_id: int
    original_text: str
    normalized_text: str
    processing_time: float
    lambda_instance_id: str

class BatchRequest(BaseModel):
    comments: List[CommentRequest]
    
    @validator('comments')
    def validate_comments(cls, v):
        if not v:
            raise ValueError('Comments list cannot be empty')
        if len(v) > 100:  # Reasonable limit for batch processing
            raise ValueError('Too many comments (max 100)')
        return v

class BatchResponse(BaseModel):
    results: List[CommentResponse]

def invoke_lambda_function(comment: CommentRequest) -> Dict[str, Any]:
    """Invoke Lambda function for a single comment"""
    logger.info(f"Invoking Lambda for comment {comment.comment_id}")
    
    try:
        # Validate input before sending to Lambda
        if not comment.text or len(comment.text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        # Prepare payload for Lambda
        payload = {
            'comment_id': comment.comment_id,
            'text': comment.text
        }
        
        # Invoke Lambda function with timeout
        response = lambda_client.invoke(
            FunctionName=settings.lambda_function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        
        if response['StatusCode'] == 200:
            # Check if response is in API Gateway format
            if 'body' in response_payload:
                # Parse the body from API Gateway response
                body_content = json.loads(response_payload['body'])
                logger.info(f"Lambda completed successfully for comment {comment.comment_id}")
                return body_content
            else:
                # Direct Lambda response
                logger.info(f"Lambda completed successfully for comment {comment.comment_id}")
                return response_payload
        else:
            error_msg = f"Lambda invocation failed with status {response['StatusCode']}: {response_payload}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
    except ValueError as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Lambda invocation error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def invoke_lambda_concurrent(comment: CommentRequest, results: List[Dict], index: int):
    """Invoke Lambda function concurrently and store result"""
    try:
        result = invoke_lambda_function(comment)
        results[index] = result
        logger.info(f"✅ Lambda completed for comment {comment.comment_id}")
    except Exception as e:
        logger.error(f"❌ Lambda failed for comment {comment.comment_id}: {str(e)}")
        results[index] = {
            'comment_id': comment.comment_id,
            'original_text': comment.text,
            'normalized_text': f"Error: {str(e)}",
            'processing_time': 0,
            'lambda_instance_id': 'error'
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/normalize", response_model=CommentResponse)
async def normalize_comment(comment: CommentRequest):
    """Normalize a single comment using Lambda function"""
    try:
        start_time = time.time()
        
        # Invoke Lambda function
        print("before invoking Lambda function", comment.comment_id)
        result = invoke_lambda_function(comment)
        print("after invoking Lambda function", result)
        
        processing_time = time.time() - start_time

        print("result", CommentResponse(
            comment_id=result['comment_id'],
            original_text=result['original_text'],
            normalized_text=result['normalized_text'],
            processing_time=processing_time,
            lambda_instance_id=result['lambda_instance_id']
        ))
        
        return CommentResponse(
            comment_id=result['comment_id'],
            original_text=result['original_text'],
            normalized_text=result['normalized_text'],
            processing_time=processing_time,
            lambda_instance_id=result['lambda_instance_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/normalize-batch", response_model=BatchResponse)
async def normalize_batch_comments(batch_request: BatchRequest):
    """Normalize multiple comments concurrently using Lambda functions"""
    try:
        start_time = time.time()
        
        # Initialize results list
        results = [None] * len(batch_request.comments)
        threads = []
        
        # Create threads for concurrent Lambda invocations
        for i, comment in enumerate(batch_request.comments):
            thread = threading.Thread(
                target=invoke_lambda_concurrent,
                args=(comment, results, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Convert results to CommentResponse objects
        response_results = []
        for result in results:
            if result:
                response_results.append(CommentResponse(
                    comment_id=result['comment_id'],
                    original_text=result['original_text'],
                    normalized_text=result['normalized_text'],
                    processing_time=result['processing_time'],
                    lambda_instance_id=result['lambda_instance_id']
                ))
        
        logger.info(f"✅ Batch processing completed in {total_time:.2f}s")
        
        return BatchResponse(results=response_results)
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Text Normalization API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "single": "/normalize",
            "batch": "/normalize-batch",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 