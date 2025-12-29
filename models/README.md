# YOLOv8 Model Placeholder

## 📦 Model Information

**Model:** YOLOv8 Custom-trained for Pest Detection  
**File:** `best.pt`  
**Size:** ~22.5 MB  
**Classes:** 4 pest types (Mosquito, Termite, Cockroach, Rodent)

---

## 🚀 How to Add the Model

### **Option 1: Copy from Local**
```bash
copy "g:\RAG\models\best.pt" "g:\RAG\Portfolio_Github\models\"
```

### **Option 2: Download from Hugging Face**
If model is too large for GitHub (>100MB), upload to Hugging Face Model Hub:

1. Create account at https://huggingface.co
2. Create new model repository
3. Upload `best.pt`
4. Update code to download from HF:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/eastern-services-yolov8",
    filename="best.pt"
)
```

### **Option 3: Git LFS (Large File Storage)**
For files >100MB:

```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git add models/best.pt
git commit -m "Add YOLOv8 model"
```

---

## 📊 Model Performance

- **Training Dataset:** 1,200+ images
- **Classes:** Termite, Bed Bug, Cockroach, Mosquito, Rat, Ant, Fly
- **Accuracy:** 92%+
- **Inference Time:** < 5 seconds

---

**Note:** Add `best.pt` file here before pushing to GitHub!
