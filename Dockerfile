FROM python:3.10.6-slim

# Install app
COPY . /usr/app
WORKDIR /usr/app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Add script and make it executable
RUN chmod +x start.sh

# Run the script
CMD ["./start.sh"]
