import requests

# Path to your test image
test_image_path = 'C:/Users/moksh/OneDrive/Desktop/Filter Identification/Testing images/images.jpg'

# Send POST request
url = 'http://127.0.0.1:5000/predict'
with open(test_image_path, 'rb') as img:
    files = {'file': img}  # The key should match what is expected in the Flask app
    response = requests.post(url, files=files)

# Print the response from the server
print(response.json())
