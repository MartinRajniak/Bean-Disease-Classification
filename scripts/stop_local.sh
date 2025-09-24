#!/bin/bash
echo "🛑 Stopping local ML environment..."

docker-compose -f docker-compose.local.yml down

echo "✅ Environment stopped. Data preserved in ./local_data/"
echo "💡 Restart anytime with: ./scripts/start_local.sh"
