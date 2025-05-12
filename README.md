# Freyberg Model Emulator

Dashboard script + everything required to put it into a Docker container. Relies on a pre-processed response matrix that's been made Sparse in order to optimize memory useage.

## Steps to Run with Python on Local Machine
1. Run the script app.py
   
2. To view it on your local machine, go to "http://http://127.0.0.1:8050/" in any web-browser

## Steps to Run with Docker on Local Machine:
1. From the parent directory, use the command "docker build -t freyberg-app ." to build Docker container
  
2. Use command "docker run -p 8050:8050 freyberg-app" to run the Dashboard locally

3. To view it on your local machine, go to "http://http://127.0.0.1:8050/" in any web-browser
