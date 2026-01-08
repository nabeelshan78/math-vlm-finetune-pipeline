# How to Run in "Offline Mode" (Local Model)
If you have downloaded the PaliGemma model to your local drive (e.g., to avoid re-downloading them every time), follow these 3 steps to point the pipeline to your local files.

## Step 1: Point to Local Model (finetune_hub/config.py)
Open finetune_hub/config.py. Change Line 13 to point to your local folder instead of the Hugging Face ID.

```python
@dataclass
class ModelConfig:
    # 1. CHANGE THIS LINE: Point to your local model folder path
    model_id: str = "models/paligemma-3b-pt-224" 
```

## Step 2: Force Offline Loading (finetune_hub/engine.py)
Open finetune_hub/engine.py. You need to add local_files_only=True to the two loading functions. This prevents the code from trying to connect to the internet.

Edit 1 (Around Line 41 - Loading Model):

```python
self.model = PaliGemmaForConditionalGeneration.from_pretrained(
    self.cfg.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True  # <--- ADD THIS LINE
)
```

Edit 2 (Around Line 50 - Loading Processor):

```python
self.processor = AutoProcessor.from_pretrained(
    self.cfg.model_id, 
    local_files_only=True  # <--- ADD THIS LINE
)
```

## Step 3: Force Offline Inference (finetune_hub/inference.py)
If you are running inference locally, you must tell the InferenceEngine to strictly use the local model files. Open finetune_hub/inference.py and modify the __init__ method.

Edit 1 (Around Line 28 - Loading Processor):

```python
self.processor = AutoProcessor.from_pretrained(
    base_model_id, 
    local_files_only=True  # <--- ADD THIS LINE
)
```

Edit 2 (Around Line 32 - Loading Base Model):

```python
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True  # <--- ADD THIS LINE
)
```



---


# Prepare Environment
1. Create a Virtual Environment
Open your terminal/command prompt and run:

Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Dependencies
Ensure you have the requirements.txt file in your folder, then install the required libraries:
```bash
pip install -r requirements.txt
```