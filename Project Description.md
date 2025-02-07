<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Smoking Detection System</title>
</head>
<body>

<h1>Real-Time Smoking Detection System</h1>

<p><strong>Developed by:</strong> <a href="mailto:magnedinanevesdina@gmail.com">Magne Dina Neves</a></p>
<p>Detect smoking behavior in real-time using cutting-edge computer vision and machine learning techniques.</p>

<hr>

<h2>📋 Overview</h2>
<p>Welcome to the Real-Time Smoking Detection System! This innovative project leverages computer vision and machine learning to detect smoking behavior by analyzing the proximity of the <strong>hand</strong> to the <strong>face</strong> in real-time. The system combines <strong>OpenCV</strong>, <strong>CVZone</strong>, and <strong>TensorFlow</strong> to create a robust and efficient solution for smoking detection.</p>

<h3>🌟 Features:</h3>
<ul>
    <li>🚀 <strong>Real-Time Detection:</strong> Instantly detects face and hand movements with minimal latency.</li>
    <li>🧠 <strong>Machine Learning Integration:</strong> Uses a Convolutional Neural Network (CNN) trained to recognize smoking behavior.</li>
    <li>🔊 <strong>Vocal Alerts:</strong> Provides real-time vocal alerts using text-to-speech technology to inform users about smoking behavior.</li>
    <li>💻 <strong>Multithreading:</strong> Ensures smooth and uninterrupted video feed while processing detection and vocal alerts concurrently.</li>
    <li>📏 <strong>Distance Calculation:</strong> Analyzes the distance between hand and face landmarks to accurately identify smoking gestures.</li>
    <li>🖥️ <strong>User-Friendly Interface:</strong> Displays clear messages: <strong>Smoking</strong> or <strong>No Smoking</strong>, along with bounding boxes around detected faces and hands.</li>
    <li>🔧 <strong>Customizable and Extendable:</strong> Easily extend and customize the system to detect other behaviors or integrate additional features.</li>
</ul>

<h3>🛠️ Innovations:</h3>
<ul>
    <li>🔍 <strong>Machine Learning for Smoke Detection:</strong> I utilized machine learning to detect images of people smoking, enhancing accuracy and reliability.</li>
    <li>🌡️ <strong>Infrared Feature:</strong> I modified my PC camera to include an infrared feature and a thermal sensor, making the detection more reliable, especially under varying lighting conditions.</li>
    <li>💡 <strong>Enhanced Accuracy:</strong> Combining traditional computer vision techniques with machine learning for improved detection accuracy and robustness.</li>
    <li>📢 <strong>Vocal Notifications:</strong> Adding a layer of user interaction by providing vocal notifications, making the system more user-friendly and engaging.</li>
</ul>

<h2>📝 How It Works:</h2>
<ol>
    <li><strong>Data Collection:</strong> I collected and labeled images of smoking and non-smoking instances to create a dataset.</li>
    <li><strong>Model Training:</strong> I trained a CNN model using TensorFlow/Keras to distinguish between smoking and non-smoking behaviors.</li>
    <li><strong>Detection:</strong> I utilized OpenCV and CVZone to detect faces and hands in real-time video frames.</li>
    <li><strong>Proximity Analysis:</strong> Calculating the distance between hand and face landmarks to detect smoking gestures.</li>
    <li><strong>Vocal Alerts:</strong> Generating real-time vocal alerts based on the detection results to inform users.</li>
</ol>

<h2>🚀 Getting Started:</h2>
<h3>Prerequisites:</h3>
<ul>
    <li>🐍 Python 3.7+</li>
    <li>🖥️ Visual Studio Code (VS Code)</li>
    <li>📦 Virtual Environment (venv)</li>
</ul>

<h3>Installation:</h3>
<ol>
    <li><strong>Clone the Repository:</strong></li>
    <pre><code>git clone https://github.com/your-repo/real-time-smoking-detection.git
cd real-time-smoking-detection
    </code></pre>
    <li><strong>Create and Activate Virtual Environment:</strong></li>
    <pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
    </code></pre>
    <li><strong>Install Dependencies:</strong></li>
    <pre><code>pip install -r requirements.txt
    </code></pre>
</ol>

<h3>Usage:</h3>
<ol>
    <li><strong>Prepare Dataset:</strong> Organize your dataset with labeled folders for smoking and non-smoking instances.</li>
    <li><strong>Train the Model:</strong> Run the training script to train the CNN model:</li>
    <pre><code>python smoke_detection_train.py
    </code></pre>
    <li><strong>Run the Detection System:</strong> Use the real-time detection script to start the system:</li>
    <pre><code>python smoke_detection_realtime.py
    </code></pre>
</ol>

![No smoking](https://github.com/user-attachments/assets/0c159e91-7f40-411e-877f-6b008f422288)

![Smoking detected](https://github.com/user-attachments/assets/7e2a0252-a9e7-4d18-90cc-20a166becd5b)
![Heat detected](https://github.com/user-attachments/assets/6f8e7666-522e-4ce5-bde4-ac77725c3965)


<h2>📧 Contact:</h2>
<p>For any questions or further information, feel free to reach out:</p>
<ul>
    <li><strong>Email:</strong> <a href="mailto:magnedinanevesdina@gmail.com">magnedinanevesdina@gmail.com</a></li>
</ul>

<h2>🚬 YOLOv5-Based Cigarette Detection</h2>
<p>In addition to the main project, I also worked on a cigarette detection system using YOLOv5.</p>

<h3>Why YOLOv5?</h3>
<p>YOLOv5 uses a deep convolutional neural network to detect objects in images or videos. It can detect objects of different sizes and aspect ratios with high accuracy and speed. YOLOv5 achieves this by using a single-stage detection framework that simultaneously predicts bounding boxes and class probabilities for each object.</p>
<p>One of the key features of YOLOv5 is its architecture, which is based on a smaller and more efficient backbone network called CSPDarknet. This results in faster training and inference times compared to other state-of-the-art object detection models.</p>
<p>Another advantage of YOLOv5 is its flexibility and ease of use. It comes with a user-friendly command-line interface that allows users to easily train and deploy their own object detection models. Additionally, it is compatible with multiple deep learning frameworks, including PyTorch, TensorFlow, and ONNX.</p>

<h3>The Dataset</h3>
<p>The dataset for this task was obtained from Kaggle, as this particular problem largely involves privacy matters and concerns. While the goal was to obtain data from real-world scenarios, the dataset provides representation close to that of actual samples, with the target object, a cigarette, appearing at many different scales and sizes.</p>
<p>Dataset link & Credit: <a href="https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection">Kaggle - Cigarette Smoker Detection</a></p>

<p>The split used:</p>
<ul>
    <li>Training: 1400 images</li>
    <li>Validation: 300 images</li>
    <li>Testing: 296 images</li>
</ul>
<p>The images were annotated in the YOLO format (.txt) using labelimg.</p>

<h3>The The Model</h3>
<p>YOLOv5s is the smallest and fastest variant of the YOLOv5 family, which makes it suitable for real-time applications on low-end devices.</p>
<p>The YOLOv5s model consists of a total of 87 layers, including 83 convolutional layers and 4 pooling layers. The model has a total of 7.6 million parameters.</p>
<p><strong>Here's a breakdown of the layer types and parameters in YOLOv5s:</strong></p>
<ul>
    <li>Convolutional layers: 83</li>
    <li>Pooling layers: 4</li>
    <li>Batch normalization layers: 84</li>
    <li>Activation layers: 83</li>
    <li>Linear layers: 3</li>
</ul>
<p><strong>The number of parameters in each type of layer is as follows:</strong></p>
<ul>
    <li>Convolutional layers: 7.4 million</li>
    <li>Batch normalization layers: 166,656</li>
    <li>Linear layers: 40,864</li>
</ul>
<p>The total number of parameters in the YOLOv5s model is therefore 7.6 million.</p>

<h3>Experimentations</h3>
<p>The model was trained using the ultralytics implementation of YOLOv5, which makes it easier to train and deploy these models. Using the available training setup, a total of 4 (Major and few minor ones, with small changes) models were tested. (Note that the image size used was 512x512)</p>
<p><strong>They are as follows:</strong></p>
<ul>
    <li><strong>Model 1:</strong> First model, therefore the Baseline for this task. With standard setup, it was trained for 10 epochs with a batch size of 16.</li>
    <li><strong>Result:</strong> It performed poorly, with a precision of 0.75542 and mAP@0.5 of 0.6785.</li>
    <li><strong>Model 2:</strong> For this run, the number of epochs was raised to 25 with batch size of 32, to give the model more examples at a time.</li>
    <li><strong>Result:</strong> It showed significant improvement, with a precision of 0.86214 and mAP@0.5 of 0.8467, but still failed to properly work while testing.</li>
    <li><strong>Model 3:</strong> Increase in epochs was a definite improvement, and so was the raise in batch size. Therefore, with the same setup, this model was trained for 50 epochs.</li>
    <li><strong>Result:</strong> It improved even further, with a precision of 0.9434 and mAP@0.5 of 0.93828, and also worked well with testing. But some other objects were also getting misclassified as a 'cigarette', this model still needed to learn more.</li>
    <li><strong>Model 4:</strong> The final run, with 100 epochs and batch size of 32.</li>
    <li><strong>Result:</strong> A lot of the issues with earlier models were addressed with this one. There were certainly some misclassifications but overall it worked best. Precision: 0.9729 and mAP@0.5: 0.9714.</li>
</ul>

<h3>Deployment</h3>
<p>The best model weights were obtained from the training process, and the inference engine was deployed on a web application using the Django backend framework with a simple UI.</p>
<p><strong>Advantages of Django:</strong></p>
<ul>
    <li>Highly scalable</li>
    <li>Secure</li>
    <li>Built-in Authentication</li>
    <li>Built-in ORM system</li>
    <li>Simple to integrate with frontend templates</li>
</ul>

<h3>Features:</h3>
<ul>
    <li>Upload Samples (Images/Videos) and store them in the database</li>
    <li>Supports real-time detections using RTSP links for CCTV footage</li>
    <li>View Samples</li>
    <li>Run Model Inferences</li>
</ul>

<h2>🚀 Getting Started:</h2>
<h3>Prerequisites:</h3>
<ul>
    <li>🐍 Python 3.7+</li>
    <li>🖥️ Visual Studio Code (VS Code)</li>
    <li>📦 Virtual Environment (venv)</li>
</ul>

<h3>Installation:</h3>
<ol>
    <li><strong>Clone the Repository:</strong></li>
    <pre><code>git clone https://github.com/your-repo/real-time-smoking-detection.git
cd real-time-smoking-detection
    </code></pre>
    <li><strong>Create and Activate Virtual Environment:</strong></li>
    <pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
    </code></pre>
    <li><strong>Install Dependencies:</strong></li>
    <pre><code>pip install -r requirements.txt
    </code></pre>
</ol>

<h3>Usage:</h3>
<ol>
    <li><strong>Prepare Dataset:</strong> Organize your dataset with labeled folders for smoking and non-smoking instances.</li>
    <li><strong>Train the Model:</strong> Run the training script to train the CNN model:</li>
    <pre><code>python smoke_detection_train.py
    </code></pre>
    <li><strong>Run the Detection System:</strong> Use the real-time detection script to start the system:</li>
    <pre><code>python smoke_detection_realtime.py
    </code></pre>
</ol>

<h2>📧 Contact:</h2>
<p>For any questions or further information, feel free to reach out:</p>
<ul>
    <li><strong>Email:</strong> <a href="mailto:magnedinanevesdina@gmail.com">magnedinanevesdina@gmail.com</a></li>
</ul>

<hr>

<p><strong>Developed by:</strong> <a href="mailto:magnedinanevesdina@gmail.com">Magne Dina Neves</a></p>
<p>This project was developed entirely by me, utilizing my skills in computer vision, machine learning, and Python programming. I am proud to present this innovative solution for real-time smoking detection.</p>

</body>
</html>
