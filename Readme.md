# Task

This repo hosts the pseudocode for KIE information extraction setup. 

## Setup 

### Create Environment
```
python3 -m venv .venv
or
python -m venv .venv
```
### Install Deps
```
pip install -r requirements.txt
```

### Docker Build Command 

```
docker build -t kie-application .
docker run -p 8000:8000 kie-application
```

## Repo Structure 

data
src

## Tasks Supported
    1. Loading Data 
        a. csv support
        b. PyTorch Data Loader 
        
    2. Classification
        a. Binary
        b. Multi-label 
    3. Entity Tagging 
    

