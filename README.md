# phyai-blenderproc
This repository, phyai-blenderproc, provides tools for generating and visualizing YOLO (You Only Look Once) format datasets using BlenderProc. It's designed to create synthetic image datasets with corresponding bounding box annotations for 3D models, which can be highly beneficial for training object detection models when real-world data is scarce or expensive to acquire.

## Quick Start
### Installation
```
conda create -n blenderproc python=3.12
conda activate blenderproc
pip install -r requirements.txt
```

### Run
- YOLO dataset generator  
`blenderproc run generate_yolo_dataset.py`

- Visualize to verify the YOLO dataset
`python visualize_yolo_dataset.py --all`


## Reference
- [BlenderProc Github](https://github.com/DLR-RM/BlenderProc) 
- [BlenderProc2 IO](https://dlr-rm.github.io/BlenderProc) 
