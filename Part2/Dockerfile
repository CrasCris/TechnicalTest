# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# update pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runner image starts here
FROM python:3.10-slim as runner

# Copy installed Python packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# set correct timezone
RUN ln -fs /usr/share/zoneinfo/America/Bogota /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

WORKDIR /part2
COPY . .

# launch api
CMD ["streamlit", "run", "app.py","--server.enableXsrfProtection","false"]