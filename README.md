# PSPNet


Link: https://arxiv.org/abs/1612.01105

### This is a pytorch replicate of PSPnet

Running on single GPU:
```bash
python main.py
```

### Goal
To test performance of PSPNet on Aerial Semantic Segmentation Drone (ASSD) Dataset

Link: https://www.kaggle.com/bulentsiyah/semantic-drone-dataset

You have to unzip it first and move children folders out of 'dataset' folder.

### Project structure

```
|-- PSPNet
    |-- _data
        |-- assd
            |-- label_images_semantic
            |-- original_images
            |-- RGB_color_image_masks
            |-- class_dict_seg.csv
    |-- _result
        |-- model
            |-- demo.pkl
    |-- _demo_img
    |-- datasets
        |-- __init__.py
        |-- dataloader_assd.py
    |-- model
        |-- layers.py
        |-- metrics.py
        |-- pspnet.py
        |-- resnet.py
    |-- demo.py
    |-- epoch.py
    |-- main.py
    |-- README.md
    |-- utils.py
    
```