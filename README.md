# BolivianMoneyDetector
System to detect coins and banknotes of the bolivian currency, focused on sight accesibility help, with the assistance of a fine-tuned YOLO v.8 Model and also with an auxiliary OCR model, and with the extracted data, to be interpreted by an LLM.

![BolivianMoneyDetectorII-39](https://github.com/user-attachments/assets/d0720784-9d48-4925-96e1-7a2f5e4699dd)

Here are the dependencies and libraries needed to run this project:
- opencv_python
- easyocr
- matplotlib
- re
- numpy
- transformers
- huggingface_hub
- ultralytics
- functools
- PIL
- pydantic
- TTS
- gradio
- PyTorch

here's the list of the commands to install a bunch of these dependencies:
```
!pip install numpy==1.23
!pip install --upgrade scipy
!pip install TTS
!pip install easyocr
!pip install opencv-python
!pip install gradio
!pip install matplotlib
!pip install torch
!pip install torchvision
!pip install huggingface_hub
!pip install transformers
!pip install fastapi
!pip install ultralytics
!pip install functools
!pip install pydantic
```
The project is configured mainly to run on MacOS, on torch.device configured on 'mps', if you have an NVIDIA graphics card and the CUDA library configured on your system, please be free to change the 'mps' value to 'cuda' and torch.backend.mps to cuda on line 172 of `detector.py` file inside /src.

#### Run Local FastAPI
To run de api locally you have to use the following command:
```
fastapi dev src/api.py
```

#### Gradio
If you need to execute the gradio UI to test the program, go to the gradio_ux.py file, execute it as a  'Run Python File' and go to the link provided on its terminal.

