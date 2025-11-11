🩺 Pulmonary Image Deep Learning Analysis Framework

📖 Project Overview (Introduction)

This project provides a comprehensive deep learning framework for advanced analysis of medical images (such as CT), including lung segmentation, lung-age prediction, data analysis and Paper image drawing.

Core Objective: 

Realizing the prediction of lung age

Core Components: Includes data preprocessing (get_lungmask.py), model definition (model.py), and the training pipeline (train_test.py).

🚀 Quick Start

1. Installation

This project recommends using Python 3.12 and setting up a virtual environment for dependency management.

Step 1: Clone the Repository

git clone [[https://github.com/YourUsername/YourProjectName.git](https://github.com/YourUsername/YourProjectName.git](https://github.com/t-zyyy/LA-Net-Model/tree/master))




Step 2: Set Up Virtual Environment

# Recommended: Use conda
conda create -n lung_analysis python=3.12
conda activate lung_analysis




2. Code Structure 💾

Model training requires adhering to a specific data structure.

2.1 Directory Structure Explanation


LA-Net_code/
├── modedls/
│   ├── resnet.py
│
└── (data pre-process↓)
├── get_lungmask.py
├── lung_box.py
├── get_instance.py
│
└── (model training↓)
├── model.py
├── brain_test.py
├── train_test.py
├── setting.py
│
└── (model testing↓)
├── ce_fenduan.py
├── xin_ce.py
├── run_ce.py
├── piancha.py
│
└── (analysing↓)
├── scatter.py
├── CE.py
│
└── (drawing of the thesis illustrations↓)
├── tools.py


2.2 Python file interpretation

In the “data pre-process” step, get_lungmask.py extracts the lung mask from the raw CT scan. 
lung_box.py then crops the lung region from the raw CT based on this mask. 
Subsequently, get_instance.py uniformly acquires slices.

In the “model training” process, brain_test.py handles data loading, model.py and resnet.py define the model architecture, 
setting.py configures hyperparameters, and finally, model training is performed in train_test.py.

In “model testing,” ce_fenduan.py can be used to test the model. 
The other three files are test files previously used during model development and can serve as references.

In “analysing,” numerous functions for processing Excel or analyzing data are defined across two files.

The “drawing of the thesis illustrations” section contains the original source code for all thesis images.



3. Model Training 💻

Use the train_test.py script to initiate model training. All configurations are centralized in the setting.py file.

3.1 Configuring Training Parameters

Before running the training, you should review and modify the following key parameters in setting.py:

--data_root

--excel_file

--test_root

--test_file

The above are the necessary addresses for the training set and the test set.

--input_D

Modify according to the number of dataset slices

--model_depth

Select the model type




4. Model Evaluation

After training is complete, use the ce_fenduan.py to evaluate model performance. 

Please ensure that the address of the trained model is placed in sets.test_path.


5. Other Nets

In the model selection stage, we also tried several other architectures and slice settings, including ResNet10, ResNet18, ResNet50, ResNet101, 20-slice and 40-slice versions.
You can find these additional models on Hugging Face: https://huggingface.co/t-zyyy/Other_Nets/tree/main


6. License

This project is licensed under the MIT License.
