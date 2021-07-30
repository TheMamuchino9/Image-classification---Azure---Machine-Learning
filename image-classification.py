from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
from PIL import Image
import os

projectId = '5e340a34-3ee0-487b-8c55-8f6bd3fc9c08'
key = '720ec1b434f04ecabbb94e6662c61f67'
endpoint = 'https://customvisionmamuch-prediction.cognitiveservices.azure.com/'
modelName = 'image-classification-exercise'

print('Ready to classify images using the model named "{}" from Custom Vision Project Id {}.'.format(modelName, projectId))
credentials = ApiKeyCredentials(in_headers={"Prediction-key": key})
client = CustomVisionPredictionClient(endpoint=endpoint, credentials=credentials)
imageFolder = 'test-images'
testImages = os.listdir(imageFolder)

fig = plt.figure(figsize=(16, 8))

print('Classifying the {} images found in {}...'.format(len(testImages), imageFolder))
for i in range(len(testImages)):
    
    with open(os.path.join(imageFolder, testImages[i-1]), "rb") as imageData:
        classification = client.classify_image(projectId, modelName, imageData)
    
    # Retrieve the first result (higest probability) from the returned prediction
    prediction = classification.predictions[0].tag_name
    
    # Display the image with its predicted class above it
    img = Image.open(os.path.join(imageFolder, testImages[i-1]))
    a=fig.add_subplot(len(testImages)/3, 4, i+1)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(prediction)

plt.show()