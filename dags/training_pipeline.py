from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
import json

def analyze_training_response(**context):
    """Analyze the training service response"""
    ti = context['ti']
    response_str = ti.xcom_pull(task_ids='call_training_service')
    
    # Parse JSON string response
    try:
        response = json.loads(response_str) if isinstance(response_str, str) else response_str
    except json.JSONDecodeError as e:
        print(f"Failed to parse response as JSON: {e}")
        print(f"Raw response: {response_str}")
        raise

    print(f"Training service response: {response}")
    
    if response.get('status') == 'success':
        metrics = response.get('metrics', {})
        print(f"Training successful!")
        print(f"Validation accuracy: {metrics.get('val_accuracy', 0):.4f}")
        print(f"Model size: {response.get('tflite_size_mb', 0):.2f} MB")
        print(f"MLflow run: {response.get('mlflow_run_id', 'unknown')}")
        
        # Echo back training config for verification
        config = response.get('training_config', {})
        print(f"Training config used: {config}")
        
        return "Training completed successfully"
    else:
        error = response.get('error', 'Unknown error')
        print(f"Training failed: {error}")
        raise Exception(f"Training failed: {error}")

dag = DAG(
    'http_training_pipeline',
    default_args={
        'owner': 'ml-engineer',
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=2),
    },
    description='ML pipeline calling training service via HTTP',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'http', 'microservice']
)

# Prepare training payload with proper types
def prepare_training_data(**context):
    """Prepare training parameters with proper types"""
    dag_run = context['dag_run']
    config = dag_run.conf or {}
    
    # Convert to proper types here instead of in template
    training_payload = {
        "epochs": int(config.get('epochs', 5)),
        "learning_rate": float(config.get('learning_rate', 0.001)),
        "batch_size": int(config.get('batch_size', 32)),
        "experiment_name": str(config.get('experiment_name', 'http_training')),
        "run_name": f"airflow_{dag_run.run_id}"
    }
    
    print(f"Prepared training payload: {training_payload}")
    return training_payload

prepare_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag,
)

# Call training service
train_model_task = SimpleHttpOperator(
    task_id='call_training_service',
    http_conn_id='training_service',
    endpoint='/train',
    method='POST',
    headers={"Content-Type": "application/json"},
    data="{{ ti.xcom_pull(task_ids='prepare_training_data') | tojson }}",
    dag=dag,
)

analyze_task = PythonOperator(
    task_id='analyze_results',
    python_callable=analyze_training_response,
    dag=dag,
)

prepare_task >> train_model_task >> analyze_task