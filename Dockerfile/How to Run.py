# Build Docker image
docker build -t psychohistory-demo .

# Run container
docker run -it --rm psychohistory-demo

# Or run directly
python demo.py
