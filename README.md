# DetectSRH

SRH single cell detection framework

[**Project Page**](https://mlins.org/elucidate) /
[**MLiNS Lab**](https://mlins.org) /
[**OpenSRH**](https://github.com/MLNeurosurg/opensrh)

[**ELUCIDATE Repo**](https://github.com/MLNeurosurg/Elucidate) / 
[**SRH550 Annotation**](https://www.dropbox.com/scl/fo/iwiirpegnky6f12sbsdra/ACo28a5SQR9gjlGH_zx9Aow?dl=0&e=2&rlkey=npvrsi7kaovfy8rfgz7nuas5k&st=d944fm4e) / 
[**DetectSRH Repo**](https://github.com/MLNeurosurg/detectsrh) / 
[**Pretrained Weights**](https://www.dropbox.com/scl/fo/iwiirpegnky6f12sbsdra/ACo28a5SQR9gjlGH_zx9Aow?dl=0&e=2&rlkey=npvrsi7kaovfy8rfgz7nuas5k&st=d944fm4e)


## Installation

1. Clone DetectSRH github repo
    ```console
    git clone git@github.com:MLNeurosurg/detectsrh.git
    ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment:
    ```console
    conda create -n ds python=3.10
    ```
4. Activate conda environment:
    ```console
    conda activate ds
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/repo/dir>
    pip install -e .
    ```

## SRH single cell detection inference instructions (using provided checkpoint)

We provide a SRH550 pretrained model checkpoint for SRH single cell detection inference. The model checkpoint is available [here.](https://www.dropbox.com/scl/fi/g3g3t3gaxkq06x07wc3cz/elucidate_model.ckpt?rlkey=m2iqgcg7dpv5v8mrz4zu1bnli&st=7zlzocly&dl=0)

A Jupyter notebook is available at [`playgrounds/inference.ipynb`](ds/playgrounds/inference.ipynb) for interactive visualization. We also provided python script, which is preferred for batch inference. To use the python inference script:

1. Configure `eval/inference.py` with model checkpoint path and srh image directory path.
2. Change directory to `eval` and activate the conda virtual environment.
3. Use `eval/inference.py` for inference:
   ```console
   python inference.py -c=config/inference_mrcnn.yaml
   ```

## DetectSRH training / evaluation instructions

The code base uses PyTorch Lightning, to train a Mask R-CNN model on cell
annotations exported from ELUCIDATE.

To train Mask R-CNN on provided SRH550 dataset:

1. Download [SRH550 dataset](https://www.dropbox.com/scl/fi/2jkjhyf19btxkqjo1awep/srh550_public.tgz?rlkey=l416wim4ntrwd2c3npif1z6jr&st=t4mngzy4&dl=0).
2. Update the sample config file in `train/config/train_mrcnn.yaml` with desired configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train.py` to start training:
   ```console
   python train.py -c=config/train_mrcnn.yaml
   ```

To evaluate with your trained Mask R-CNN model:

1. Update the sample config files in `eval/config/eval_mrcnn.yaml` with the
   checkpoint path and other desired configurations per file.
2. Change directory to `eval` and activate the conda virtual environment.
3. Use `eval/eval.py` for evaluation:
   ```console
   python eval.py -c=config/eval_mrcnn.yaml
   ```

