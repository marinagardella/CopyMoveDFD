# CopyMoveDFD


### Usage

To run the method, simply provide the path to the input document image. The script will process the image and save the output `detection.png` showing all detected duplicated regions (exact matches up to `t'), each pair connected by a line and enclosed in bounding boxes. The output saved in the current working directory.


An example on how to run the method is given below:
```
python main.py input_document.png -t 0.8
```
Where `-t` is the proportion of exact matches required to flag two patches.

### Online demo

You can try the method online in the following <a href="https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000547">IPOL demo</a>.
