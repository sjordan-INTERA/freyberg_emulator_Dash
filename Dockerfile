FROM python:3.10

ENV DASH_DEBUG_MODE=False

# Set working directory
WORKDIR /app

# Copy app and zipped file
COPY ./app /app

# Install dependencies + unzip file
RUN set -ex && \
    apt-get update && apt-get install -y unzip && \
    pip install --no-cache-dir -r requirements.txt && \
    unzip /app/assets/response.zip -d /app/assets/ && \
    rm /app/assets/response.zip

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]

