# Face Recognition System
A real-time face recognition system using advanced models to identify and match faces in video streams.

https://github.com/user-attachments/assets/f30cc5e8-9caf-475d-957e-2ced89a1b44d

# How to use

## Create Environment and Install Packages

1. Clone the repository:
   ```bash
   git clone https://github.com/HussainNasirKhan/Face-Recognition-System.git
   cd Face-Recognition-System

2. Create new environment:
   ```bash
   conda create -n face-recog python=3.9
3. Activate new environment:
   ```bash
   conda activate face-recog
  
4. Install the required dependencies:
   ```bash
   pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   pip install -r requirements.txt

### Add new persons to datasets (new_persons folder)
1. Create a folder with the folder name being the name of the person
2. Add the person's photo in the folder
3. Run to add new persons: 
   ```python
   python add_persons.py
4. Run to recognize:
   ```python
   python recognize.py
## Tech Stack:
### Face Detection
1. Retinaface
2. Yolov5-face
3. SCRFD (Single-Shot Scale-Aware Face Detector)
### Face Recognition
1. ArcFace
### Face Tracking
1. ByteTrack
### Matching Algorithm
1. Cosine Similarity Algorithm

## Credits

This project is based on the work from [Original Repository](https://github.com/vectornguyen76/face-recognition). Thank you to the original author for his contributions.
