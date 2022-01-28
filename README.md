# Face Alignment

Face Alignment using python

Input Image | Aligned Face | Aligned Face | Aligned Face  |
--- | --- | --- | --- |
![model arch](assets/friends.jpg) | ![model arch](assets/friends_0.jpg) | ![model arch](assets/friends_2.jpg) | ![model arch](assets/friends_1.jpg) |

Input Image | Aligned Face | Input Image | Aligned Face  |
--- | --- | --- | --- |
![model arch](assets/trump.jpg) | ![model arch](assets/trump_0.jpg) | ![model arch](assets/scarlett-johansson.jpeg) | ![model arch](assets/scarlett-johansson_0.jpg) |


## Installation

Install required packages
```
pip install -r requirements.txt
```


## Inference

This code processes an image or a directory of images and save output to a directory:

```
python3 align_image.py --input ./input/friends.jpg --output ./output
python3 align_image.py --input ./input --output ./output
```

or run following command to align face image using imutils package:
```
python3 align_image_2.py --input ./input/friends.jpg --output ./output
```
