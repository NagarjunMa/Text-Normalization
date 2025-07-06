import json
import boto3
import time
import logging
import re
from typing import Dict, Any, List
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

class BedrockTextNormalizer:
    """Text normalizer using AWS Bedrock Nova LLM"""
    
    def __init__(self):
        # Nova model ID - using the correct Nova Lite model ID
        self.model_id = "amazon.nova-lite-v1:0"
        
    def create_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to normalize text while preserving numbers"""
        prompt = f"""
Please normalize the following text for an insurance underwriter. 
The text should be grammatically correct and professional while preserving all numbers, percentages, and currency values exactly as they appear.

Rules:
1. Preserve all numbers, percentages (like 80.5%), currency amounts (like $500), and ranges (like 7.5 ~ 8)
2. Fix grammar and make the text more professional
3. Keep the meaning exactly the same
4. Make it suitable for insurance underwriting documentation

Text to normalize: "{text}"

Normalized text:"""
        return prompt.strip()
    
    def extract_numbers_before_llm(self, text: str) -> dict:
        """Extract and preserve numbers before sending to LLM"""
        # Define number patterns to preserve (order matters - most specific first)
        number_patterns = [
            r'\$\d+\.?\d*\s*~\s*\$\d+\.?\d*',  # Currency ranges: $500 ~ $750
            r'\d+\.?\d*\s*~\s*\d+\.?\d*',      # Number ranges: 7.5 ~ 8
            r'\d+\.?\d*%',                      # Percentages: 15%, 7.5%
            r'\$\d+\.?\d*',                     # Currency: $2500
            r'\b\d+\.?\d*\b',                   # Regular numbers: 3.2, 48 (with word boundaries)
        ]
        
        placeholder_map = {}  # placeholder -> original_number
        modified_text = text
        processed_positions = set()
        
        print(f"Original text: '{text}'")
        
        # Process patterns from most specific to least specific
        for i, pattern in enumerate(number_patterns):
            matches = list(re.finditer(pattern, modified_text))

            
            # Process matches in reverse order to maintain positions
            for match in reversed(matches):
                print(f"Match: {match}")
                start, end = match.span()
                number_str = match.group()

                
                # Check if this position overlaps with already processed positions
                overlap = False
                for pos_start, pos_end in processed_positions:
                    if not (end <= pos_start or start >= pos_end):
                        overlap = True
                        print(f"Overlap detected with position {pos_start}-{pos_end}")
                        break
                
                if not overlap and number_str not in placeholder_map.values():
                    placeholder = f"__NUMBER_{len(placeholder_map)}__"
                    placeholder_map[placeholder] = number_str
                    processed_positions.add((start, end))
                    
                    
                    # Replace the number with placeholder
                    modified_text = modified_text[:start] + placeholder + modified_text[end:]
                else:
                    print(f"Skipping '{number_str}' (overlap or duplicate)")
        
        
        return modified_text, placeholder_map
    
    def restore_numbers_after_llm(self, text: str, placeholder_map: dict) -> str:
        """Restore preserved numbers after LLM processing using placeholder map"""
        result = text
        
        # Find all NUMBER patterns in the text
        import re
        number_pattern = re.compile(r'__NUMBER_\d+__')
        matches = number_pattern.findall(result)
        
        
        # Replace each placeholder with its original number from the map
        for placeholder in matches:
            if placeholder in placeholder_map:
                original_number = placeholder_map[placeholder]
                print(f"Replacing '{placeholder}' with '{original_number}'")
                result = result.replace(placeholder, original_number)
            else:
                print(f"Warning: Placeholder '{placeholder}' not found in map")
        
        return result
        
        return result
    
    def normalize_with_bedrock(self, text: str) -> str:
        """Normalize text using Bedrock Nova LLM"""
        try:
            # Step 1: Extract and preserve numbers
            text_with_placeholders, placeholder_map = self.extract_numbers_before_llm(text)
            
            # Step 2: Create prompt for LLM
            prompt = self.create_prompt(text_with_placeholders)
            print(f"Prompt: {prompt}") 
            
            # Step 3: Call Bedrock Nova LLM
            request_body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
            }
            
            response = bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Step 4: Parse response
            response_body = json.loads(response['body'].read())
            # Nova model returns 'completion' field
            normalized_text = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '').strip()            
            
            # Step 5: Restore numbers using the map
            final_text = self.restore_numbers_after_llm(normalized_text, placeholder_map)
            
            return final_text
            
        except ClientError as e:
            logger.error(f"Bedrock API error: {str(e)}")
            raise Exception(f"Bedrock API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in text normalization: {str(e)}")
            raise Exception(f"Text normalization error: {str(e)}")

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Lambda function handler for text normalization"""
    start_time = time.time()
    
    try:
        # Parse input
        if 'body' in event:
            # API Gateway event
            body = json.loads(event['body'])
        else:
            # Direct Lambda invocation
            body = event
        
        comment_id = body.get('comment_id')
        text = body.get('text')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Text is required'
                })
            }
        
        # Initialize normalizer
        normalizer = BedrockTextNormalizer()
        
        # Normalize text
        normalized_text = normalizer.normalize_with_bedrock(text)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response_body = {
            'comment_id': comment_id,
            'original_text': text,
            'normalized_text': normalized_text,
            'processing_time': processing_time,
            'lambda_instance_id': context.aws_request_id
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        logger.error(f"Lambda function error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        } 