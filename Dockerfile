FROM python:3.10

ENV DASH_DEBUG_MODE=False

# Set working directory
WORKDIR /app

# Copy app files
COPY ./app /app

# Install dependencies
RUN set -ex && \
    apt-get update && apt-get install -y unzip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]

