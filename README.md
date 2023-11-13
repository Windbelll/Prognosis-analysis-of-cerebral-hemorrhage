# Instructions
## Pre-requisties
- python>=3.9
- CUDA11.1+ + cuDNN
## Install dependencies
```
pip install -r requirements.txt
```
## Pre-processing
**we provide a visual way to configure YAML files, which can more intuitively understand what we have done in the pre-process, the personal information in this sample file has been hidden**

```sh
python visualize_adjust_Image.py --input_path="./sample/sample.dcm" --output_yaml_path="./settings/test.yaml"
```
then you can see the interface:

![image-20230316001610997](.\example\image_vis.png)

If you want to start processing from DICOM files:

```sh
python data_pretreatment/preprocess_format_convert.py --input_path=YOUR_DIR -y
```

DICOM files will convert to PNG files in "./data_pretreatment/outputs/"


-----

then, pre-process the image:

```sh
python data_pretreatment/preprocess_skull_division.py --input_path=YOUR_DIR -y --as_dir
```

output files in "./data_pretreatment/outputs_skull/"

"-y" represents unified management of configuration through YAML files

```yaml
threshold: 240
window_level: 35
window_width: 80
```


## Train
*tips: You need to first classify the patient into good and bad prognosis. And the patient's fold should be named "patient_name_GCS"*

Please manually fold bad and good into the data folder, then run:

```sh
python data_to_dataset.py --train_rate=80 --val_rate=10 --test_rate=10
```

The program will generate three txt files under the data fold.

Finally, use the following code to start training:

```sh
python check_gcs.py --batch_size=32 --epochs=300 --learning_rate=0.0001 --with_gcs=True --use_gpu=True
```

## Inference & CAM

Due the privacy of our dataset, we only provide 3 pictures for the result viewing. They are located at "./example/", You can also use your own dataset to inference and view CAM results.
```sh
python cam_test.py --path="example/sample.png" --GCS=14 --with_gcs=True
```

"--with_gcs" refers to using baseline or GCS-ICHNet

here is sample slices' outcome:

![image](.\example\sample_cam.png)

### Related Links

https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/brain-tumor-ai-challenge-2021

https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/code

https://csrankings.org/#/fromyear/2011/toyear/2023/index?all&us

https://github.com/52CV/CV-Surveys

