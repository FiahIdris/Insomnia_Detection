# Use a base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create Streamlit config directory and add theme settings
RUN mkdir -p ~/.streamlit && \
    echo "\
[theme]\n\
primaryColor = '#e44652'\n\
backgroundColor = '#faf8f0'\n\
secondaryBackgroundColor = '#e4dfcf'\n\
textColor = '#043353'\n\
font = 'monospace'\n\
" > ~/.streamlit/config.toml

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "your_script.py", "--server.port=8501", "--server.address=0.0.0.0"]
