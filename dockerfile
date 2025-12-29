# CHANGE THIS LINE: Use 3.11 instead of 3.9
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ca-certificates

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
COPY . .

# Expose Ports
EXPOSE 8000
EXPOSE 8501

# Default command
CMD ["python", "main.py"]