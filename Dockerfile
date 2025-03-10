# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app to the container
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]