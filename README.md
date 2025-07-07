# Text Normalization Tool

A professional text normalization application for insurance underwriting using AWS Bedrock Nova LLM. This tool provides a clean, modern interface for normalizing insurance comments while preserving critical numerical data.

Loom Recording - https://www.loom.com/share/ddbeef3476704744a58d86e8d353e6c5?sid=0f4dc80e-68a1-4663-96da-d450818b97ac

## 🏗️ Architecture

```
Frontend (Streamlit) → API Gateway → Lambda Function → AWS Bedrock Nova LLM
```

## 🚀 Features

- **Professional Text Normalization**: Converts informal comments into professional insurance documentation
- **Number Preservation**: Maintains all numerical values, percentages, and currency amounts exactly
- **Batch Processing**: Process multiple comments concurrently for improved efficiency
- **Real-time Processing**: Fast response times with AWS Lambda and Bedrock
- **Modern UI**: Clean, responsive interface built with Streamlit

## 📋 Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS CLI configured
- Docker (optional, for containerized deployment)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd text-normalize
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials
```bash
aws configure
```

### 4. Deploy AWS Resources
Follow the detailed deployment guide in [DEPLOYMENT.md](./DEPLOYMENT.md)

## 🏃‍♂️ Quick Start

### Local Development
```bash
# Start the API server
python api_server_gateway.py

# In another terminal, start the frontend
streamlit run app.py
```

### Production Deployment
```bash
# Deploy Lambda function
./scripts/deploy-lambda.sh

# Deploy API Gateway
./scripts/deploy-api-gateway.sh

# Start the application
python app.py
```

## 📁 Project Structure

```
text-normalize/
├── app.py                      # Streamlit frontend application
├── api_server_gateway.py       # FastAPI backend server
├── lambda_function.py          # AWS Lambda function
├── requirements.txt            # Python dependencies
├── Dockerfile.frontend         # Frontend container
├── Dockerfile.backend          # Backend container
├── scripts/                    # Deployment scripts
│   ├── deploy-lambda.sh
│   └── deploy-api-gateway.sh
├── tests/                      # Test suite
│   ├── test_lambda.py
│   ├── test_api.py
│   └── test_normalizer.py
├── docs/                       # Documentation
│   ├── API.md
│   └── DEPLOYMENT.md
└── config/                     # Configuration files
    ├── lambda-role-policy.json
    └── api_gateway_config.yaml
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Test Lambda function
python -m pytest tests/test_lambda.py -v

# Test API endpoints
python -m pytest tests/test_api.py -v

# Test text normalization
python -m pytest tests/test_normalizer.py -v
```

## 📊 API Documentation

### Endpoints

- `POST /normalize` - Normalize a single comment
- `POST /normalize-batch` - Normalize multiple comments
- `GET /health` - Health check endpoint

### Request Format
```json
{
  "comment_id": 1,
  "text": "Loan-to-value high. Need bring down to 80.5%. Risk too big."
}
```

### Response Format
```json
{
  "comment_id": 1,
  "original_text": "Loan-to-value high. Need bring down to 80.5%. Risk too big.",
  "normalized_text": "The loan-to-value ratio is high. We need to bring it down to 80.5%. The risk is too significant.",
  "processing_time": 1.234,
  "lambda_instance_id": "lambda-123"
}
```

## 🔧 Configuration

### Environment Variables
- `API_ENDPOINT`: Backend API server endpoint
- `AWS_REGION`: AWS region for Lambda and Bedrock
- `LAMBDA_FUNCTION_NAME`: Name of the deployed Lambda function

### AWS Configuration
- Ensure Bedrock access is enabled in your AWS account
- Configure appropriate IAM roles and policies
- Set up API Gateway with proper CORS settings

## 🚀 Performance

- **Single Comment**: ~1-2 seconds
- **Batch Processing**: Concurrent processing for multiple comments
- **Lambda Cold Start**: ~500ms
- **Bedrock Response**: ~800ms average

## 🔒 Security

- Input validation and sanitization
- AWS IAM role-based access control
- API Gateway authentication (configurable)
- Secure environment variable handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the [documentation](./docs/)
- Review the [deployment guide](./DEPLOYMENT.md)

## 🏆 Acknowledgments

- AWS Bedrock for LLM capabilities
- Streamlit for the beautiful UI framework
- FastAPI for the robust API framework 