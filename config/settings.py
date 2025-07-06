"""
Configuration management for the text normalization application
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    lambda_function_name: str = Field(default="text-normalization-lambda", env="LAMBDA_FUNCTION_NAME")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_endpoint: str = Field(default="http://localhost:8000", env="API_ENDPOINT")
    
    # Lambda Configuration
    lambda_timeout: int = Field(default=30, env="LAMBDA_TIMEOUT")
    lambda_memory_size: int = Field(default=256, env="LAMBDA_MEMORY_SIZE")
    
    # Bedrock Configuration
    bedrock_model_id: str = Field(default="amazon.nova-lite-v1:0", env="BEDROCK_MODEL_ID")
    bedrock_temperature: float = Field(default=0.7, env="BEDROCK_TEMPERATURE")
    bedrock_top_p: float = Field(default=0.9, env="BEDROCK_TOP_P")
    
    # Application Configuration
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Security Configuration
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def validate_aws_credentials() -> bool:
    """Validate AWS credentials are configured"""
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    return all(os.getenv(var) for var in required_vars)


def get_lambda_arn() -> str:
    """Get Lambda function ARN"""
    account_id = os.getenv("AWS_ACCOUNT_ID")
    if not account_id:
        raise ValueError("AWS_ACCOUNT_ID environment variable is required")
    
    return f"arn:aws:lambda:{settings.aws_region}:{account_id}:function:{settings.lambda_function_name}"


def get_api_gateway_url() -> Optional[str]:
    """Get API Gateway URL if configured"""
    return os.getenv("API_GATEWAY_URL")


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    log_level: str = "DEBUG"
    cors_origins: list = ["http://localhost:3000", "http://localhost:8501"]


class ProductionSettings(Settings):
    """Production environment settings"""
    log_level: str = "WARNING"
    enable_auth: bool = True
    cors_origins: list = Field(default_factory=list)  # Must be explicitly set


def get_environment_settings() -> Settings:
    """Get environment-specific settings"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    else:
        return DevelopmentSettings() 