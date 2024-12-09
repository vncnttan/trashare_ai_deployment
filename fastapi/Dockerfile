# Use the official Python base image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /

# Install the Python dependencies
RUN pip install fastapi uvicorn tensorflow==2.11.0 numpy==1.21.6 Pillow requests python-multipart

# Copy the application code to the container
COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
