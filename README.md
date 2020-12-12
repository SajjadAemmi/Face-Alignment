# Face Alignment


#### Requirements
Python3 with dlib,numpy,scipy and pillow.

### To run

This code processes a directory of images (this directory should only contain images) 
and outputs corresponding aligned images to another directory.
```
python3 align_images.py --src_dir /source_directory --out_dir /output_directory
```

This code processes an image and output corresponding aligned image

```
python3 align_images.py --input /path/to/input-image --output /path/to/output-image`
```