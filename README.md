# DR_API

Diabetic retinopathy prediction API. It uses Python and is built using the TensorFlow library. It uses the ResNet model for the training of the model. The trained model is saved as a .h5 file and is later used in the API to predict the result

## How to run the API

To run the API one should install all the required libraries and modules into the local system or virtual environment. After which in the terminal of your computer or the virtual environment (choose where you have installed the required files) write the following command:

_**uvicorn --reload main:app**_

After which the API will start on the specified port
