FROM python:3.10-slim

WORKDIR /app

# Copy the Python dependencies file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ src/

# Run the application on container startup
CMD ["python", "src/main.py"]