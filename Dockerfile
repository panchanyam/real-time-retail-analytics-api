# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project files into container
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]