# Use an official PyTorch runtime with GPU support as the base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update -y \
    && apt-get -o Dpkg::Options::="--force-confmiss" install --reinstall netbase -y \
    && apt-get install libgl1-mesa-glx -y \
    && apt-get install libglib2.0-0  -y

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports
EXPOSE 5000

# Define the command to run the model
CMD ["python",  "app.py"]
