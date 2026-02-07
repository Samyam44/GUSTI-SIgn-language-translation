# Deployment Guide

## Quick Start

### 1. Installation

```bash
# Clone or download the application
cd asl-recognition-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup

Place your trained models in the application directory:

```
asl-recognition-app/
├── app.py
├── sentence_model.h5          # Sentence recognition model
├── sentence_labels.txt         # Sentence class labels
├── alphabet_model.h5           # Alphabet recognition model
├── alphabet_labels.txt         # Alphabet class labels
├── realtime_engine.py
├── translation_tts.py
├── webcam_capture.py
└── templates/
    └── index.html
```

**Labels File Format** (one label per line):
```
Hello
Thank you
I love you
...
```

### 3. Running the Application

```bash
# Basic run
python app.py

# With custom host/port
python app.py --host 0.0.0.0 --port 8080
```

Access the application at: `http://localhost:5000`

### 4. Testing

```bash
# Run test suite
python test_app.py
```

## Production Deployment

### Using Gunicorn (Recommended for Linux/macOS)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app

# With better configuration
gunicorn -w 4 \
  -b 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile access.log \
  --error-logfile error.log \
  --log-level info \
  app:app
```

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

Build and run:

```bash
# Build image
docker build -t asl-recognition .

# Run container
docker run -p 5000:5000 --device /dev/video0 asl-recognition
```

### Using Nginx as Reverse Proxy

Create `/etc/nginx/sites-available/asl-app`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # WebSocket support for video streaming
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/asl-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Configuration Options

### Environment Variables

Create `.env` file:

```bash
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Model Paths
SENTENCE_MODEL_PATH=sentence_model.h5
SENTENCE_LABELS_PATH=sentence_labels.txt
ALPHABET_MODEL_PATH=alphabet_model.h5
ALPHABET_LABELS_PATH=alphabet_labels.txt

# Camera Configuration
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Detection Configuration
SMOOTHING_WINDOW_SIZE=5
CONFIDENCE_THRESHOLD=0.6

# Translation Configuration
DEFAULT_LANGUAGE=es

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

Load in `app.py`:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Use environment variables
SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
SENTENCE_MODEL_PATH = os.getenv('SENTENCE_MODEL_PATH', 'sentence_model.h5')
```

### Performance Tuning

**For High FPS:**
```python
SMOOTHING_WINDOW_SIZE = 3  # Faster response
CONFIDENCE_THRESHOLD = 0.5  # More predictions
```

**For High Accuracy:**
```python
SMOOTHING_WINDOW_SIZE = 7  # More stable
CONFIDENCE_THRESHOLD = 0.7  # Fewer false positives
```

**GPU Acceleration:**
```python
# Add to app.py initialization
import tensorflow as tf

# Use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"Using GPU: {gpus[0]}")
```

## Monitoring and Logging

### Setup Application Logging

```python
# In app.py
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
file_handler = RotatingFileHandler(
    'app.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

app.logger.addHandler(file_handler)
```

### Health Check Endpoint

Add to `app.py`:

```python
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    health_status = {
        'status': 'healthy',
        'sentence_engine': sentence_engine is not None,
        'alphabet_engine': alphabet_engine is not None,
        'webcam': webcam is not None and webcam.is_opened(),
        'translator': translator is not None
    }
    
    all_healthy = all(health_status.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify(health_status), status_code
```

Monitor with:
```bash
curl http://localhost:5000/health
```

## Security Considerations

### 1. HTTPS/SSL

Use Let's Encrypt for free SSL:

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Rate Limiting

Install Flask-Limiter:

```bash
pip install Flask-Limiter
```

Add to `app.py`:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/set_engine', methods=['POST'])
@limiter.limit("10 per minute")
def set_engine():
    # ... existing code
```

### 3. CORS Configuration

For API access from different domains:

```bash
pip install flask-cors
```

```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://your-frontend-domain.com"]
    }
})
```

## Troubleshooting

### Common Issues

**1. Webcam Access Denied**
```bash
# Linux: Add user to video group
sudo usermod -a -G video $USER

# Then logout and login
```

**2. Out of Memory Errors**
```python
# Reduce model batch size or use smaller models
# In realtime_engine.py:
model_predictions = self.model.predict(features, verbose=0, batch_size=1)
```

**3. Slow Performance**
- Reduce camera resolution: 320x240 instead of 640x480
- Increase confidence threshold
- Use GPU if available
- Reduce smoothing window size

**4. Translation API Errors**
- Check internet connection
- Implement retry logic
- Use cached translations for common phrases

### Logs Location

```bash
# Application logs
tail -f app.log

# System logs (Ubuntu)
journalctl -u asl-app -f

# Nginx logs
tail -f /var/log/nginx/error.log
```

## Systemd Service (Linux)

Create `/etc/systemd/system/asl-app.service`:

```ini
[Unit]
Description=ASL Recognition Application
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/asl-recognition-app
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable asl-app
sudo systemctl start asl-app
sudo systemctl status asl-app
```

## Backup and Recovery

### Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/asl-app"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz \
  *.h5 *.txt

# Backup config
cp .env $BACKUP_DIR/env_$DATE

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

## Performance Benchmarks

Expected performance on different hardware:

| Hardware | FPS | Latency | Notes |
|----------|-----|---------|-------|
| CPU only (i5) | 10-15 | 100-150ms | Acceptable |
| CPU + GPU (GTX 1660) | 25-30 | 30-50ms | Recommended |
| CPU + GPU (RTX 3070) | 30+ | 20-30ms | Optimal |

## Support and Maintenance

- Check logs regularly
- Monitor system resources
- Update dependencies monthly
- Retrain models with new data
- Backup configurations and models
- Test after any changes

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
