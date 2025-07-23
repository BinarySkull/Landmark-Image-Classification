# 1. Use an official Python runtime as a parent image
# Using the slim version to keep the image size down
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
# We point to the CPU-only PyTorch repository to avoid installing the large CUDA version
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# 5. Copy your application code and model into the container
COPY main.py .
COPY transfer_exported.pt .

# 6. Expose the port the app will run on
EXPOSE 80

# 7. Define the command to run your application
# We use --host 0.0.0.0 to make the container accessible from the network
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]