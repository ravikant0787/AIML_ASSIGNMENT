# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy app.py, Dockerfile, requirements.txt, and the mlruns folder
# from the root of the Hugging Face Space repository into the /app directory.
COPY app.py .
COPY Dockerfile .
COPY requirements.txt .
COPY mlruns ./mlruns/ # Copy mlruns to /app/mlruns

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Flask app will run on
EXPOSE 5000

# Command to run the Flask application using Gunicorn
# app.py will be executed from /app, and mlruns is accessible at /app/mlruns
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
