# Master thesis  AnomalyDiffusion

## Data and checkpoints

The generated anomaly data and all the checkpoints can be downloaded from the following links. And some checkpoints should be placed 
at the corresponding directory.
(Note that we have filtered out some data with poor generation effects. Therefore, some classes
have relatively fewer samples.)

[//]: # (You can download the checkpoints for the UNet models trained on the generated data from )

[//]: # ([Google Drive]&#40;https://drive.google.com/drive/folders/1kcOdfQrvWeJyliGTYJ4HXKU5ccfn7t96?usp=sharing&#41;)

[//]: # (or [百度网盘]&#40;https://pan.baidu.com/s/1Xoe__ODeq_YrVc9lA-7B_A&#41; &#40;提取码: 0306&#41;.)

[//]: # (|                                          | Google Drive | 百度网盘 |)

[//]: # (|------------------------------------------|--------------|----------|)

[//]: # (| Generated data                           |         [Google Drive]&#40;https://drive.google.com/file/d/1yzsZdW_xS-v4GprE2KQmQ1EbIWyGyFcG/view?usp=sharing&#41;     |    [百度网盘]&#40;https://pan.baidu.com/s/12gKMfc64sy3JDx5FAR-ytQ&#41; &#40;提取码: 0306&#41;     |)

[//]: # (| Checkpoints for anomaly generation model |              |          |)

[//]: # (| Checkpoints for mask generation model    |              |          |)

[//]: # (| Checkpoints for anomaly localization     |     [Google Drive]&#40;https://drive.google.com/drive/folders/1kcOdfQrvWeJyliGTYJ4HXKU5ccfn7t96?usp=sharing&#41;         |    [百度网盘]&#40;https://pan.baidu.com/s/1Xoe__ODeq_YrVc9lA-7B_A&#41; &#40;提取码: 0306&#41;      |)

[//]: # (| Checkpoints for anomaly classification   |              |          |)

| Data and Models                          | Download                                                                                             | Place at                  |
|------------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------|
| Generated data                           | [Google Drive](https://drive.google.com/file/d/1fV2S-Memcll0oAnrPmfNLgi8E7yb7XTC/view?usp=drive_link)   | $path_to_the_generated_data                          |
| Checkpoints for anomaly generation model | [Google Drive](https://drive.google.com/drive/folders/17SA6QWGH4Mxk4lTIDm2DpG0N3PcpWicl?usp=sharing) | logs/anomaly-checkpoints  |                                                                                   |
| Checkpoints for mask generation model    | [Google Drive](https://drive.google.com/drive/folders/1LPJCd2dwocPHnA-Ex6d9aHFVk1JGHZ7Q?usp=sharing) | logs/mask-checkpoints     |
| Checkpoints for anomaly localization     | [Google Drive](https://drive.google.com/drive/folders/1PYq1I00JBij9J7IvNdYsQWLFnY0eQ20v?usp=sharing) | checkpoints/localization  |
| Checkpoints for anomaly classification   | [Google Drice](https://drive.google.com/drive/folders/1XhSaDZJQb9d6VYkf5GU3C8a4XgjGfB0N?usp=sharing)                                                                                     | checkpoints/classification |

## Overview
Anomalydiffusion is a few-shot anomaly generation model for anomaly inspection (detection, localization and classification). 
The overall process can be divided into the following 5 steps:

(1) Train the anomaly generation model and mask generation model;

(2) Generate anomalous masks by the mask generation model in step (1);

(3) Generate anomaly mask-image pairs by the anomaly generation model based on the generated masks in step (2);

(4) Train the anomaly detection (for both detection and localization) and classification model based on the anomalous image-mask pairs in step (3).



## Prepare


### (1) Prepare the environment
```
Ubuntu
python 3.8
cuda==11.8
gcc==7.5.0
conda env create -f environment.yaml
conda activate Anomalydiffusion
```


### (2) Checkpoint for LDM

Download the official checkpoint of the latent diffusion model:
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Generating anomalous image-mask pairs

In this section, you can first train the anomaly generation model by (1). After that, you can run (2), which
contains training mask generation models, generating anomalous masks and generating anomalous image-mask pairs.

### (1) Train the anomaly generation model by:

```
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --spatial_encoder_embedding --data_enhance
 --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t 
 --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0, 
  --init_word anomaly  --mvtec_path=$path_to_mvtec_dataset
```

### (2) Train the mask generation model and generate image-mask pairs by:
```
CUDA_VISIBLE_DEVICES=$gpu_id python run-mvtec.py --data_path=$path_to_mvtec_dataset
```
## Test the anomaly inspection performance

[//]: # (### &#40;1&#41; Generating anomlay image-mask pairs)

[//]: # (After training &#40;or downloading&#41; the anomalous generation model and mask generation model,)

[//]: # (you can generate anomaly masks first, and then generate anomalous image-mask pairs.)

[//]: # ()
[//]: # (To generate **anomaly masks**, you can run:)

[//]: # ()
[//]: # ()
[//]: # (After generating anomalous masks, you can generate **anomalous image-mask paris** by:)

After generating anomalous image-mask pairs,
you can train and test the **anomaly detection model** (for both anomlay detection and localization) by:
```
python train-localization.py --generated_data_path $path_to_the_generated_data  --mvtec_path=$path_to_mvtec
python test-localization.py --generated_data_path $path_to_mvtec
```

you can train and test the **anomaly classification model** by:
```
python train-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data
python test-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data
```

## Compute IC-LPIPS
To compute IC-LPIPS for the generated dataset, please run:
```
python cal_ic_lpips.py --mvtec_path=$path_to_mvtec --gen_path=$path_to_the_generated_data
```


