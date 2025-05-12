# Freyberg Model Emulator

Dashboard script + everything required to put it into a Docker container

## Steps to Run:
1. Unzip the response matrix from 'response.zip' file into /app/assets/master 

2. From the parent directory, use the command "docker build -t freyberg-app ." to build Docker container
  
3. Use command "docker run -p 8050:8050 freyberg-app" to run the Dashboard locally

4. To view it on your local machine, go to "http://http://127.0.0.1:8050/" in any web-browser
