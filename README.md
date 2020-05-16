# Helmet-Detection-using-CNN
To overcome the issues and significant drawbacks of earlier proposed works, we try to develop a better, faster, and more robust framework. We recommend a method that uses deep learning [17] and OpenCV [18] to solve this problem. First, we train the network using a dataset containing annotated images of road traffic. We label the photos with three classes: 
  HelmetNo (two-wheelers without a helmet)
  Helmet (two-wheelers with a motorcycle helmet)
  NumberPlate (plates of vehicles)
Using these labels, we recognize the classes and then identify the plates when necessary.
The key idea is to use object detection for the system. To achieve this, we use the CNN model. We generate a feature vector for each region of the image using CNN. These vectors form a tensor for further processing of image data. 
We use the YOLO algorithm for performing training and detection. YOLO uses a Convolution Neural Network for Object Detection. When we provide an image or video input, the model is used to detect the above classes. We separate the regions comprising commuters without a helmet and apply number plate detection to those regions. OCR Space does optical character recognition on the number plates detected to extract the characters.
