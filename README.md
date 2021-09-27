# Face Alignment

Face Alignment using python

Input Image | Aligned Face | Aligned Face | Aligned Face  |
--- | --- | --- | --- |
![model arch](input/friends.jpg) | ![model arch](output/friends_0.jpg) | ![model arch](output/friends_2.jpg) | ![model arch](output/friends_1.jpg) |

Input Image | Aligned Face | Input Image | Aligned Face  |
--- | --- | --- | --- |
![model arch](input/trump.jpg) | ![model arch](output/trump_0.jpg) | ![model arch](input/scarlett-johansson.jpeg) | ![model arch](output/scarlett-johansson_0.jpg) |


## Installation

Install required packages
```
pip install -r requirements.txt
```


## Inference

This code processes an image and output to a directory:

```
python3 align_image.py --input ./input/friends.jpg --output ./output
```

or run following command to align face image using imutils package:
```
python3 align_image_2.py --input ./input/friends.jpg --output ./output
```
