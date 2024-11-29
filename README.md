# OpenCV-Object-Detection <img src="https://skillicons.dev/icons?i=python"/>
Real time object detection using Computer Vision and the OpenCV library
Overview

This project leverages Active Machine Learning (AML), Augmented Reality (AR), and Convolutional Neural Networks (CNNs) to create an interactive web application for teaching object recognition and sign language to children with Autism Spectrum Disorders (ASD). By addressing the unique learning challenges of children with ASD, this solution aims to foster engagement, improve communication, and provide a personalized learning experience.
Features
1. Object Recognition

    Interactive Learning: Teaches children to identify objects grouped by categories and difficulty levels.
    Real-Time Detection: Uses a webcam to capture and process video frames for object recognition.
    Multimedia Learning: Presents information through text, images, audio, video, and 3D objects.

2. Sign Language Learning

    Flip Cards: Allows children to learn and practice sign language through interactive flashcards.
    Matching Exercises: Matches signs with their corresponding words or pictures.
    Inclusivity: Provides alternative communication methods for non-verbal children.

3. Customization and Self-Assessment

    Personalized Learning: Adjusts learning pace and content to individual needs.
    Progress Tracking: Includes a test page for self-assessment and learning progress monitoring.

Technical Stack
Preprocessing

    RGB to Grayscale Conversion: Simplifies image data.
    Resizing: Standardizes image dimensions.
    Gabor Filter: Reduces noise and enhances critical features.
    Region Proposal Network (RPN): Segments images to identify object regions.

Feature Extraction

    Gray-Level Co-occurrence Matrix (GLCM): Extracts texture features (contrast, energy, homogeneity).

Model Training

    Convolutional Neural Network (CNN):
        Built using TensorFlow or PyTorch.
        Trains on preprocessed images to classify objects.
        Validated through cross-validation or test datasets.

Real-Time Object Identification

    Webcam Integration: Captures video frames for processing.
    Object Classification: Uses the trained CNN model for real-time object recognition.
    Text-to-Speech (TTS): Converts object names into speech for auditory feedback.
