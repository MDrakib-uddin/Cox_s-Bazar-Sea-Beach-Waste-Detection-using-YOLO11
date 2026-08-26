# Plastic Waste Detection with YOLO

COCO-format Cox's Bazar beach plastic-waste dataset YOLO11 training project.

## Project files

- `yolo-plastic-waste-training.ipynb`: interactive Kaggle/Jupyter workflow
- `train_yolo.py`: complete conversion, training, validation and `.pt` export script
- `requirements.txt`: Python dependencies
- `dataset/`: original COCO dataset from https://data.mendeley.com/datasets/bdzg4tjy63/1m 

## Classes

The script ignores the unused COCO placeholder class `objects` and trains on the 14 annotated classes: bag, bottle, bottle_cap, cup, fishing_item, net, others, packet, polythene, rope, spoon, straw, sunglass and toy.

## Local setup

Use Python 3.10 or newer. From this project folder:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python train_yolo.py --epochs 50 --batch 16 --predict
```

For CPU training, use a smaller batch:

```bash
python train_yolo.py --epochs 10 --batch 4
```

The converted dataset is written to `dataset_yolo/`. Training results and the exported model are written to `runs/`.

## Kaggle

In Kaggle, add the dataset and run:

```bash
pip install -r requirements.txt
python train_yolo.py --data-root /kaggle/input/datasets --epochs 50 --batch 16 --predict
```

Kaggle input is read-only. The script automatically writes converted data and training output to:

```text
/kaggle/working/dataset_yolo
/kaggle/working/runs
```

After training, download this file from the Kaggle file browser or output:

```text
/kaggle/working/runs/plastic_waste_yolo11n_best.pt
```

## Notebook order

Run the notebook cells from top to bottom: install dependencies, locate the dataset, convert COCO annotations, create `data.yaml`, train, evaluate, predict, then save/download the `.pt` model.

## Important options

```text
--data-root   COCO root containing train, valid and test
--output-dir  converted YOLO dataset path
--runs-dir    writable training output path
--epochs      number of epochs, default 50
--batch       batch size, default 16
--imgsz       image size, default 640
--weights     starting model, default yolo11n.pt
--predict     run one test-image prediction after training
```
