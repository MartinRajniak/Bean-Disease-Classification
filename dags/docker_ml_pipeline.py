from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.python import PythonOperator

def check_training_results(**context):
    """Check training results in MLflow"""
    print("Training completed! Check MLflow UI: http://localhost:5000")
    return "Training successful"

dag = DAG(
    'ssh_docker_pipeline',
    default_args={
        'owner': 'ml-engineer',
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
    },
    description='ML pipeline using SSH to run Docker on host',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'ssh', 'docker']
)

# SSH task to run training on host
train_model_task = SSHOperator(
    task_id='train_model_ssh',
    ssh_conn_id='localhost_ssh',  # We'll configure this
    command="""
    cd /home/martin/Workspace/Bean-Disease-Classification && \
    ./scripts/docker_run.sh src/training/train_model.py \
      "{{ dag_run.conf.get('epochs', '5') }}" \
      "{{ dag_run.conf.get('learning_rate', '0.001') }}" \
      "{{ dag_run.conf.get('batch_size', '32') }}" \
      "ssh_training"
    """,
    dag=dag,
)

analyze_task = PythonOperator(
    task_id='analyze_results',
    python_callable=check_training_results,
    dag=dag,
)

train_model_task >> analyze_task