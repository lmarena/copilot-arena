# Use an official Python runtime as a parent image
FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:0.3.0 /uv /bin/uv

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN uv pip install --system -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2", "--log-config", "log_conf.yaml"]

