# Stock-market-lstm-prediction-aws
End-to-end Stock Market LSTM Prediction System using AWS SageMaker, Lambda, API Gateway, SNS, and Power BI
![Stock market Architecture](https://github.com/user-attachments/assets/d36924ba-cafe-41c1-b179-9d75c592cc0a)

### Stock Market Prediction – Correct Architecture Explanation

1️⃣ Model Training & Deployment (Amazon SageMaker)

Historical stock market data is collected using yfinance inside a SageMaker Jupyter Notebook.

Data is preprocessed and used to train an LSTM model.

The trained model is saved to Amazon S3.

A SageMaker Endpoint is created to serve real-time predictions.

Purpose:
To build and host a scalable machine learning model.

2️⃣ Real-Time Prediction Endpoint (SageMaker Endpoint)

The SageMaker endpoint exposes the trained model as a secure HTTPS endpoint.

It accepts JSON input and returns predicted stock prices.

Purpose:
To provide real-time inference capability.

3️⃣ Backend Processing (AWS Lambda)

An AWS Lambda function is used to:

Receive HTTPS requests

Invoke the SageMaker endpoint

Process prediction responses

Lambda also triggers SNS notifications when conditions are met.

Purpose:
To handle business logic without managing servers.

4️⃣ HTTPS API Interface (Amazon API Gateway)

Amazon API Gateway exposes the Lambda function as an HTTPS endpoint.

This endpoint is:

Secured

Publicly accessible

Used for sending prediction requests


5️⃣ API Testing via Postman (Client Tool)

Postman is used to:

Send HTTPS POST requests to the API Gateway URL

Pass input data in JSON format

View prediction responses returned by Lambda

Purpose:
To validate and test the deployed HTTPS API.

6️⃣ Email Alerts (Amazon SNS)

Amazon SNS sends email notifications based on prediction results.

Lambda publishes messages to an SNS topic.

Subscribers receive alerts automatically.

Purpose:
To notify users of important prediction events.

7️⃣ Visualization (Power BI)

Power BI consumes prediction data via the HTTPS API.

Dashboards visualize predicted stock prices and trends.

Purpose:
To convert predictions into business insights.



## SageMaker Endpoint 
<img width="1920" height="1080" alt="Aws sagemaker endpoint" src="https://github.com/user-attachments/assets/5a456b83-1703-4aa1-bcf1-0560820d0e96" />

## AWS Lambda Execution Output


<img width="1920" height="1080" alt="Lambda output" src="https://github.com/user-attachments/assets/e77eef73-20c6-4269-aded-d1220f3c1ff9" />

## SNS Email Notification

<img width="1910" height="845" alt="SNS Notification" src="https://github.com/user-attachments/assets/a7e66698-32c6-413b-8354-4280a0f19c8c" />

## API Gateway – Postman Output

<img width="1920" height="1080" alt="Postman Api " src="https://github.com/user-attachments/assets/c5164926-0c57-440b-ae80-87fcab8a5de9" />

## Power BI Dashboard


<img width="1920" height="1080" alt="Powerbi multiple stock predictin" src="https://github.com/user-attachments/assets/94eee31e-cc49-4113-a307-445ab0064613" />


