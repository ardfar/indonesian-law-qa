
# Indonesian Law QA System

![App Screenshot](https://github.com/ardfar/indonesian-law-qa/blob/main/screenshot.png?raw=true)

An Question Answering system to serve answers to Indonesian law-related questions. Based on BERT architecture which has been pretrained with SQuAD dataset then re-trained on 5 most popular law topics in Indonesia such as:

- Tax
- ITE
- Legacy
- Lands
- Crypto

## Run Locally

Clone the project

```bash
  git clone https://github.com/ardfar/indonesian-law-qa.git
```

Go to the project directory

```bash
  cd indonesian-law-qa
```

Install required libraries via pip

```bash
  pip install numpy pandas torch transformer flask
```

Start the web application

```bash
  python main.py
```


## Run Locally with GPU

Clone the project

```bash
  git clone https://github.com/ardfar/indonesian-law-qa.git
```

Go to the project directory

```bash
  cd indonesian-law-qa
```

(Optional) Create Conda Environment

```bash
  conda create -n "ilq" python=3.8.0
```

Activate Conda Environment

```bash
  conda activate ilq
```

Install CUDNN and CUDA toolkit to use Nvidia GPU Inference on Conda Environment

```bash
  conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9.7 
```

Install Torch

```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install required libraries via pip

```bash
  pip install numpy pandas transformer flask
```

Start the web application

```bash
  python main.py
```
## Developers

- [Farras Arrafi](https://www.github.com/ardfar)
- Rizky Ryu
- Natalia Desy
- Ninastasya
- Ruspan Kurniadi


## Acknowledgements

This project was created as a final project of Natural Languange Processing (NLP) of AI Class

