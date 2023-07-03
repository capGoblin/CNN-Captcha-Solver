# CNN Captcha Solver

This repository contains a Convolutional Neural Network (CNN) model for solving Captcha images. The model is trained using a dataset obtained from Kaggle.

## Dataset

The dataset used for training the model can be found on Kaggle [link to dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images). It consists of a collection of Captcha images along with their corresponding labels.

## Dependencies

The following dependencies are required to run the code:

- Python 3.x
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- look at the code for more

## Network Architecture
The CNN model used for solving the Captcha images has the following architecture:
```
Input: (40, 20, 1) image
Conv2D (16 filters, 3x3 kernel, ReLU activation) -> MaxPooling2D (2x2 pool)
Conv2D (32 filters, 3x3 kernel, ReLU activation) -> MaxPooling2D (2x2 pool)
Conv2D (128 filters, 3x3 kernel, ReLU activation) -> MaxPooling2D (2x2 pool)
Conv2D (128 filters, 3x3 kernel, ReLU activation) -> MaxPooling2D (2x2 pool)
Flatten
Dense (1500 units)
Dense (19 classes, softmax)
Compiled with categorical cross-entropy loss and Adam optimizer.
```
This architecture was found to be effective in solving the Captcha images and achieved good results(kinda).

## Results
After training the model on the dataset, the following results were obtained:

#### On Training Data(at last epoch):
  - Accuracy: 100%
  - Loss: 2.0184 x 10<sup>-6</sup> or 2.0184e-06

#### On Validataion Data(at last epoch):
  - val_accuracy: 89.5%
  - val_loss: 130% or 1.3061

#### On Test Data .evaluate():
  - Accuracy: 87.8%
  - Loss: 165% or 1.6563

#### On 5 random images .predict(): 
  - 5/5

These results demonstrate the effectiveness of the trained model in solving Captcha images.
