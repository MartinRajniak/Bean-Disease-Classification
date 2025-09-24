#!/bin/bash
echo "🚀 Starting local ML development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
echo "📦 Starting containers..."
docker-compose -f docker-compose.local.yml up -d

echo "⏳ Waiting for services to start (this may take 2-3 minutes)..."
sleep 60

# Check if services are running
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow is ready!"
else
    echo "⚠️  MLflow still starting..."
fi

if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Airflow is ready!"
else
    echo "⚠️  Airflow still starting (may take another minute)..."
fi

echo ""
echo "🎉 Environment started!"
echo "🌐 Airflow UI: http://localhost:8080"
echo "👤 Login: admin / password: admin"
echo "📊 MLflow UI: http://localhost:5000"
echo "💾 Data persisted in: ./local_data/"
echo ""
echo "💡 Tip: Run 'docker-compose -f docker-compose.local.yml logs -f' to see logs"
