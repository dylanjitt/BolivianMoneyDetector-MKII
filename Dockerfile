# Start with a base image of macOS Sequoia ARM64
FROM macOS/sequoia:arm64

# Install Python 3.11.5
RUN brew install python@3.11

# Set Python 3.11.5 as the default python
RUN ln -s /opt/homebrew/opt/python@3.11/bin/python3.11 /usr/local/bin/python3 && \
    ln -s /opt/homebrew/opt/python@3.11/bin/pip3.11 /usr/local/bin/pip3

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose port 8080
EXPOSE 8080

# Set the default command to run the application using FastAPI
CMD ["fastapi", "dev", "src/api.py"]

