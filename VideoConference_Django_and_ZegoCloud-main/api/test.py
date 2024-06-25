import requests

# Replace with the path to your image file
image_file = "C:\\Users\\GOKUL RAJ\\Downloads\\VideoConference_Django_and_ZegoCloud-main\\VideoConference_Django_and_ZegoCloud-main\\api\\H25_jpg.rf.a7c599391c5578d0dcbe1eb8fff0d00a.jpg"

# Read the image file as binary data
with open(image_file, "rb") as f:
    image_data = f.read()

# Send the POST request with the image file
url = "http://localhost:8000/predict/"
files = {"image": ("image.jpg", image_data, "image/jpeg")}
response = requests.post(url, files=files)

# Check the response status code
if response.status_code == 200:
    # Successful response
    predictions = response.json()
    print("Predictions:")
    for prediction in predictions:
        print(f"Label: {prediction['label']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Bounding Box: ({prediction['xmin']}, {prediction['ymin']}), ({prediction['xmax']}, {prediction['ymax']})")
        print()
else:
    # Error response
    print(f"Error: {response.json()['message']}")