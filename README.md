# NIU
# PI-baseline

## Dataset

​	If you want to use this code, please replace the dataset URL in ./datasets/datasets.py : DEFAULT_ROOT ='Your URL '

​	If you want to replace the CUB URL, in ./datasets/CUB.py : root_path = "Your URL "

## Running the code

### **Environment**

- Python 3.7.3

- Pytorch 1.2.0

- tensorboardX

  ### Training Meta-baseline

  ```
  python train_classifier.py --config configs/train_classifier_mini.yam
  
  
  python train_meta.py --config configs/train_meta_mini.yaml
  ```

### 	Get prior knowledge

​		You should replace the load_encoder  in save_feature.yaml

```
python save_feature.py --config configs/save_feature.yaml
```

### 	Fuse the method

​		You should replace the load_encoder and save_feature in  train_meta_demo.yaml

```
python train_meta_NI.py --config configs/train_meta_mini_demo.yaml
```

### 		Test

​		You should replace the load_encoder, save_feature and number_class in  test_few_shot.yaml

```
python test_few_shot.py --config configs/test_few_shot.yaml --shot 1
```

## Acknowledgments

This code is based on the implementations of  Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning
