# Python runtime
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "GUI streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
