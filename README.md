# A-Hybrid-Approach-to-Human-Activity-Recognition

This project focuses on detecting human violence using a hybrid deep learning method that leverages both spatial and temporal features. It uses MobileNetV2 for extracting spatial features from individual frames and ConvLSTM for capturing temporal patterns across video sequences. The entire system is organized into four key modules.

The solution is divided into four key modules:

**Module 1:Model Workbook**

In this module, we develop and train the hybrid model using a dataset containing both violent and non-violent video clips. MobileNetV2 is used for feature extraction, while ConvLSTM processes the sequential data to learn the temporal dynamics of the activities. We employ the ReLU activation function and use early stopping to prevent overfitting during training. The model achieves a high accuracy of 96% and is saved in .h5 format for further use in the following modules. 

**Dataset Link -** https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

(**Note:** To make integration easier, the trained model.h5 file is also included directly in the repository named **modelnew.h5**.

Also in module 3 and 4 you have to replace the Telegram bot token and chat ID to get notified.Those credentials is unique to every user that should be handle with safety and security measures.


**Module 2: Violence Detection Model**

Here, we implement the logic to load and use the trained model for real-time or uploaded video analysis. This module checks for the presence of violent activity in the video input. If violence is detected, it clearly displays the result, helping users or systems respond appropriately to the detected behavior.

**Module 3: Violence Alert System**

This module is responsible for alert generation. When violent activity is detected in Module 2, the system captures an image of the individual involved, along with the date, time, and location of the event. It then sends an instant alert via a Telegram bot to notify the concerned authorities or officials. This ensures timely action and monitoring of such incidents.

**Module 4: Web Application Integration (Streamlit)**

In the final module, we bring together all the previous components into a single, user-friendly web application built with Streamlit. The user can upload a video through the interface. The system processes the video to detect any violent activity, and if detected, automatically triggers the alert system. This provides an end-to-end solution for real-time human activity monitoring and violence detection.

(Note: Save the module 4 in single word say final.py In the command prompt go to the folder where all modules are saved and do run the command **python -m streamlit run final.py**)
