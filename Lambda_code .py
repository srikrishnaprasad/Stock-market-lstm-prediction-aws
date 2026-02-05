import os
import json
import boto3
import base64
import traceback
from datetime import datetime
import csv
from io import StringIO

# Environment Variables
SCALER_BUCKET = os.environ["SCALER_BUCKET"]
SCALER_KEY = os.environ["SCALER_KEY"]
ENDPOINT_NAME = os.environ["SM_ENDPOINT"].strip()
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]

# AWS Clients
REGION = "ap-south-1"
s3 = boto3.client("s3", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)

# Global Cache
SCALER_PARAMS = None


def get_stock_name():
    """
    Auto-detect stock name from S3 metadata
    Checks S3 on EVERY invocation (no caching) to always get latest stock
    """
    # Try to read from S3 metadata file
    try:
        metadata_key = "stock_lstm/stock_metadata.json"
        print(f"🔍 Checking metadata at s3://{SCALER_BUCKET}/{metadata_key}")
        obj = s3.get_object(Bucket=SCALER_BUCKET, Key=metadata_key)
        metadata = json.loads(obj["Body"].read())
        stock_name = metadata.get("stock_name", "UNKNOWN")
        print(f"✅ Stock name from metadata: {stock_name}")
        return stock_name
    except s3.exceptions.NoSuchKey:
        print(f"⚠️ No metadata file found at stock_lstm/stock_metadata.json")
    except Exception as e:
        print(f"⚠️ Error reading metadata: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Fallback to environment variable or default
    stock_name = os.environ.get("STOCK_NAME", "UNKNOWN")
    print(f"📊 Using stock name: {stock_name} (from env or default)")
    return stock_name


def load_scaler_params():
    """Load MinMaxScaler parameters from S3 (cached)"""
    global SCALER_PARAMS
    if SCALER_PARAMS is None:
        print(f"📥 Loading scaler from s3://{SCALER_BUCKET}/{SCALER_KEY}")
        obj = s3.get_object(Bucket=SCALER_BUCKET, Key=SCALER_KEY)
        SCALER_PARAMS = json.loads(obj["Body"].read())
    return SCALER_PARAMS


def inverse_minmax(pred, scaler):
    """Inverse MinMax scaling to get real prices"""
    return [
        (pred[i] / scaler["data_scale"][i]) + scaler["data_min"][i]
        for i in range(len(pred))
    ]


def parse_input(event):
    """Parse API Gateway event body"""
    body = event.get("body", event)

    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()

    if isinstance(body, str):
        body = json.loads(body)

    if "instances" not in body:
        raise ValueError("Missing 'instances' in request")

    return body["instances"]


def save_to_s3(prediction_data, stock_name):
    """
    Save predictions to both JSON and CSV for Power BI
    Now includes stock name in all records
    """
    now = datetime.utcnow()
    timestamp = now.isoformat()
    
    result = {
        "stock_name": stock_name,
        "timestamp": timestamp,
        "prediction_date": now.strftime("%Y-%m-%d"),
        "prediction_time": now.strftime("%H:%M:%S"),
        "open_price": prediction_data["Open price"],
        "close_price": prediction_data["Closing price"],
        "price_change": round(
            prediction_data["Closing price"] - prediction_data["Open price"], 2
        ),
        "price_change_percent": round(
            ((prediction_data["Closing price"] - prediction_data["Open price"]) 
             / prediction_data["Open price"]) * 100, 2
        )
    }
    
    csv_updated = False
    json_saved = False
    archived = False
    
    try:
        # 1. Save latest JSON
        s3.put_object(
            Bucket=SCALER_BUCKET,
            Key="predictions/latest_prediction.json",
            Body=json.dumps(result, indent=2),
            ContentType="application/json"
        )
        json_saved = True
        print(f"✅ Saved latest_prediction.json")
        
    except Exception as e:
        print(f"❌ Failed to save JSON: {str(e)}")
        traceback.print_exc()
    
    try:
        # 2. Update historical CSV using proper CSV writer
        csv_key = "predictions/historical_predictions.csv"
        
        # Try to read existing CSV
        existing_rows = []
        try:
            print(f"📊 Reading existing CSV from {csv_key}")
            obj = s3.get_object(Bucket=SCALER_BUCKET, Key=csv_key)
            csv_content = obj['Body'].read().decode('utf-8')
            
            # Parse existing CSV
            reader = csv.DictReader(StringIO(csv_content))
            existing_rows = list(reader)
            print(f"📊 Found {len(existing_rows)} existing records")
            
        except s3.exceptions.NoSuchKey:
            print(f"📄 CSV not found, will create new file")
            
        except Exception as e:
            print(f"⚠️ Error reading CSV: {str(e)}, creating new file")
        
        # Add new row
        existing_rows.append(result)
        
        # Write CSV using proper csv.writer
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header (includes stock_name as first column)
        writer.writerow([
            'stock_name',
            'timestamp',
            'prediction_date',
            'prediction_time',
            'open_price',
            'close_price',
            'price_change',
            'price_change_percent'
        ])
        
        # Write all rows
        for row in existing_rows:
            writer.writerow([
                row.get('stock_name', stock_name),
                row['timestamp'],
                row['prediction_date'],
                row['prediction_time'],
                row['open_price'],
                row['close_price'],
                row['price_change'],
                row['price_change_percent']
            ])
        
        # Get CSV content
        csv_content = output.getvalue()
        
        # Save to S3
        print(f"💾 Saving CSV with {len(existing_rows)} records")
        s3.put_object(
            Bucket=SCALER_BUCKET,
            Key=csv_key,
            Body=csv_content.encode('utf-8'),
            ContentType='text/csv'
        )
        csv_updated = True
        print(f"✅ CSV updated successfully! Total records: {len(existing_rows)}")
        
    except Exception as e:
        print(f"❌ Failed to update CSV: {str(e)}")
        traceback.print_exc()
    
    try:
        # 3. Archive individual JSON
        archive_key = f"predictions/archive/{now.strftime('%Y-%m-%d')}/{timestamp}.json"
        s3.put_object(
            Bucket=SCALER_BUCKET,
            Key=archive_key,
            Body=json.dumps(result, indent=2),
            ContentType="application/json"
        )
        archived = True
        print(f"✅ Archived: {archive_key}")
        
    except Exception as e:
        print(f"❌ Failed to archive: {str(e)}")
        traceback.print_exc()
    
    return {
        "json_saved": json_saved,
        "csv_updated": csv_updated,
        "archived": archived
    }


def send_sns(stock_name, open_p, close_p, price_change_pct):
    """Send SNS email alert with prediction"""
    try:
        message = (
            f"📈 Stock Market Prediction Alert\n\n"
            f"Stock: {stock_name}\n"
            f"Predicted Open Price: ₹{open_p:.2f}\n"
            f"Predicted Close Price: ₹{close_p:.2f}\n"
            f"Expected Change: {price_change_pct:+.2f}%\n\n"
            f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"📈 {stock_name} LSTM Prediction",
            Message=message
        )
        print("✅ SNS alert sent")
        return True
        
    except Exception as e:
        print(f"⚠️ SNS failed: {str(e)}")
        return False


def lambda_handler(event, context):
    """Main Lambda handler for stock price prediction"""
    
    print(f"🚀 Lambda invoked - Request ID: {context.aws_request_id}")
    
    # Auto-detect stock name from S3 metadata (checked every time)
    stock_name = get_stock_name()
    
    print(f"📊 Stock: {stock_name}")
    print(f"📍 Endpoint: {ENDPOINT_NAME}")
    print(f"📦 Bucket: {SCALER_BUCKET}")
    
    try:
        # 1. Load scaler parameters
        scaler = load_scaler_params()
        
        # 2. Parse input instances
        instances = parse_input(event)
        print(f"📊 Received {len(instances)} instance(s)")
        
        # 3. Invoke SageMaker endpoint
        print(f"🤖 Invoking SageMaker endpoint: {ENDPOINT_NAME}")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"instances": instances})
        )
        
        # 4. Parse predictions
        preds = json.loads(response["Body"].read())["predictions"]
        print(f"✅ Received {len(preds)} prediction(s)")
        
        # 5. Inverse transform to real prices
        results = []
        for p in preds:
            real = inverse_minmax(p, scaler)
            results.append({
                "Open price": round(real[0], 2),
                "Closing price": round(real[1], 2)
            })
        
        print(f"💰 {stock_name} Prediction: Open={results[0]['Open price']}, Close={results[0]['Closing price']}")
        
        # 6. Save to S3 for Power BI (JSON + CSV)
        s3_status = save_to_s3(results[0], stock_name)
        
        # 7. Send SNS alert
        price_change_pct = (
            (results[0]["Closing price"] - results[0]["Open price"]) 
            / results[0]["Open price"]
        ) * 100
        sns_sent = send_sns(
            stock_name,
            results[0]["Open price"], 
            results[0]["Closing price"],
            price_change_pct
        )
        
        # 8. Return success response with stock name
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "stock_name": stock_name,
                "predicted_prices": results,
                "timestamp": datetime.utcnow().isoformat(),
                "s3_status": s3_status,
                "sns_sent": sns_sent
            })
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": str(e),
                "request_id": context.aws_request_id
            })
        }
