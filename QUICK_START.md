# Baby Language Model - Quick Start 🚀

## ❓ Your Questions Answered

### Q: Is it using RNN?
**A: YES!** It uses **LSTM (Long Short-Term Memory)**, which is a type of RNN.

The model has **2 LSTM layers** (both are RNNs):
```python
layers.LSTM(128, return_sequences=True),  # ← RNN layer 1
layers.LSTM(128),                         # ← RNN layer 2
```

### Q: How do I run it in main?
**A: Three ways! Pick one:**

---

## 🏃 How to Run

### Option 1: Simplest (One Command)
```bash
python baby_language_model.py
```
That's it! The file has a `main()` function at the bottom that runs automatically.

---

### Option 2: Quick Script (Recommended)
```bash
python simple_run.py
```

This file contains:
```python
from baby_language_model import BabyLanguageModel, BABY_DATASET

baby = BabyLanguageModel(BABY_DATASET, seq_length=15)
baby.build_model(lstm_units=64)
baby.train(epochs=150)

# Generate baby talk!
print(baby.generate_text("me want", length=50))
```

---

### Option 3: Interactive (Your Own Script)

Create `my_baby_test.py`:
```python
from baby_language_model import BabyLanguageModel, BABY_DATASET

# Step 1: Create model
baby = BabyLanguageModel(BABY_DATASET)

# Step 2: Build the RNN
baby.build_model(lstm_units=64, dropout=0.2)

# Step 3: Train
baby.train(epochs=150, batch_size=32)

# Step 4: Generate
print(baby.generate_text("me want", length=50, temperature=1.0))
print(baby.generate_text("me love", length=50, temperature=1.0))

# Step 5: Save
baby.save_model('my_trained_baby.keras')
```

Then run:
```bash
python my_baby_test.py
```

---

### Option 4: In Python Interactive Shell
```bash
python
```
Then:
```python
>>> from baby_language_model import main
>>> baby = main()  # Trains and generates

# Or step by step:
>>> from baby_language_model import BabyLanguageModel, BABY_DATASET
>>> baby = BabyLanguageModel(BABY_DATASET)
>>> baby.build_model()
>>> baby.train(epochs=100)
>>> baby.generate_text("me want", length=50)
```

---

### Option 5: Google Colab (No Setup!)
1. Go to https://colab.research.google.com
2. New notebook
3. First cell:
```python
!pip install tensorflow numpy
```
4. Second cell - copy entire `baby_model_colab.py` file
5. Run!

---

## 📦 Required Packages

```bash
pip install tensorflow numpy
```

Or if on Apple Silicon Mac:
```bash
pip install tensorflow-macos tensorflow-metal numpy
```

---

## 🎯 What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `baby_language_model.py` | Main model class | Import this in your code |
| `simple_run.py` | Minimal example | Quick test |
| `run_baby_model.py` | Complete examples | Learn different methods |
| `baby_model_colab.py` | Google Colab version | Use in Colab |
| `baby_transformer_model.py` | Transformer version | Advanced users |
| `BABY_MODEL_GUIDE.md` | Full documentation | Read for details |
| `ARCHITECTURE_EXPLAINED.md` | RNN explanation | Understand the model |

---

## 🧪 Quick Test (30 seconds)

```bash
# Install
pip install tensorflow numpy

# Run
python simple_run.py

# Done! You'll see baby talk generated!
```

---

## 🔧 Common Issues

### "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "No module named 'baby_language_model'"
Make sure you're in the correct directory:
```bash
cd /Users/soroushebadi/Desktop/Projects/ML_learning
python simple_run.py
```

### Model generates garbage
- Train longer: `baby.train(epochs=300)`
- More data: Add more sentences to `BABY_DATASET`
- Lower temperature: `baby.generate_text("me want", temperature=0.7)`

---

## 💡 Quick Customization

### Add Your Own Baby Talk
Edit `baby_language_model.py` line 13:
```python
BABY_DATASET = [
    "me want cookie",
    "me love puppy",
    # Add yours here!
    "me like dinosaurs",
    "me go playground",
]
```

### Change Model Size
```python
baby.build_model(
    lstm_units=128,  # Bigger = smarter but slower (default: 64)
    dropout=0.2
)
```

### Train More
```python
baby.train(
    epochs=300,      # More = better (default: 150)
    batch_size=32
)
```

### Generate More Text
```python
baby.generate_text(
    seed_text="me want",
    length=100,        # Longer output
    temperature=0.8    # 0.5=safe, 1.0=normal, 1.5=creative
)
```

---

## 🎓 Understanding the Output

### During Training
```
Epoch 1/150
32/32 [======] - 2s 45ms/step - loss: 2.845 - accuracy: 0.234
```
- **loss**: Lower is better (aim for <0.5)
- **accuracy**: Higher is better (aim for >0.8)

### After Training
```
me want cookie me love mommy me play toy
```
- Generates character by character
- Uses RNN memory to maintain context
- Temperature controls creativity

---

## 🚀 Next Steps After Running

1. **Experiment with temperature**:
   ```python
   baby.generate_text("me want", temperature=0.5)  # Conservative
   baby.generate_text("me want", temperature=1.5)  # Creative
   ```

2. **Add more data**: Expand `BABY_DATASET` to 100+ sentences

3. **Save and reuse**:
   ```python
   baby.save_model('my_baby.keras')
   # Later:
   baby.load_model('my_baby.keras')
   ```

4. **Try different prompts**:
   ```python
   for prompt in ["me want", "me love", "daddy", "mommy"]:
       print(baby.generate_text(prompt, length=40))
   ```

---

## 📞 Need Help?

- Read `BABY_MODEL_GUIDE.md` for detailed instructions
- Read `ARCHITECTURE_EXPLAINED.md` to understand RNN
- Check `run_baby_model.py` for more examples

---

**Ready? Run this now:**
```bash
python simple_run.py
```

**That's all you need!** 🎉
