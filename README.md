# Segment Anything with OpenVino

This proj is a demo to run segment anything on OpenVino which can make use of Intel GPU acceleration.

You can running demo through:
 1. sam.ipynb
 2. Setup following env and run locally

## Installation Basic Dependencies

Python
```
winget install -e --id Python.Python.3.10
```
PIP
```
python -m ensurepip
```
GIT
```
winget install -e --id Git.Git
```

## Installation SAM Pipeline on OpenVino

Environment preparation
First, please follow below method to prepare your development environment, you can choose download model from HuggingFace for better runtime experience.

```
$ pip install opencv-contrib-python
$ pip install openvino openvino-dev onnx
$ pip install torch==1.13.1 #important
$ pip install segment_anything
$ pip install gradio>=3.25
```
SAM Model url: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 

## Usage

1. Model Conversion
This step convert the orignal models to OpenVino optimized IR format and can run with Intel GPU acceleration.
```
$ python sam_model_convert.py
```
Note: above convertion script needs to set proper path of sam_vit_b_01ec64.pth. Pls modify sam_model_convert.py code properly for sam model path.

After running convertion. Please check your current path, make sure you already generated below models currently. Other ONNX files can be deleted for saving space.

sam_image_encoder.<xml|bin>
sam_mask_predictor.<xml|bin>
* If your local path already exists ONNX or IR model, the script will jump tore-generate ONNX/IR. If you updated the pytorch model or want to generate model with different shape, please remember to delete existed ONNX and IR models.

2. Runtime pipeline test for SAM to segment orig img
Prepare input image: orignal image. (orignal.png in repo)
Note: in this repo, above image example is provided.

Start running local server:

```
python segmentAnything.py
```

After start executing segmentAnything.py, a local host server is started.
Open the web page http://127.0.0.1:7860/
Pull the orignal image. Click a postion to generate it's corresponding contour through SAM model running in OpenVino.
In parallel, mask images are generated.
sam_mask_result_selected.png
sam_mask_result_reverted.png
These masks can be sent to SD+inpaint to redraw content outside/inside the contour.


## Disclaimer

This software is designed to contribute positively to the AI-generated media industry, assisting artists with tasks like character animation and models for clothing.

We are aware of the potential ethical issues and have implemented measures to prevent the software from being used for inappropriate content, such as nudity.

Users are expected to follow local laws and use the software responsibly. If using real faces, get consent and clearly label deepfakes when sharing. The developers aren't liable for user actions.


## Licenses

Our software uses a lot of third party libraries as well pre-trained models. The users should keep in mind that these third party components have their own license and terms, therefore our license is not being applied.


## Credits

- [deepinsight](https://github.com/deepinsight) for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models.
- all developers behind the libraries used in this project
