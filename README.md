<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1>Real-Time Smoking Detection System</h1>

<p><strong>Developed by:</strong> <a href="mailto:magnedinanevesdina@gmail.com">Magne Dina Neves (Krypton)</a></p>
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
    <li>🌡️ <strong>Infrared Feature:</strong> I modified my PC camera to include an infrared feature, making the detection more reliable, especially under varying lighting conditions.</li>
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

