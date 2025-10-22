# Baby Language Model - RNN Architecture Explained 🧠

## Yes, It Uses RNN! 🎯

The baby language model uses **LSTM (Long Short-Term Memory)**, which is a type of **RNN (Recurrent Neural Network)**.

---

## Visual Architecture

```
INPUT TEXT: "me want"
     ↓
┌────────────────────────────────────────┐
│  Character Sequence: ['m','e',' ','w'] │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  Convert to Numbers (Indices)          │
│  [13, 5, 1, 23]                        │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  LAYER 1: Embedding (32 dimensions)    │
│  Maps each number to a vector          │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  LAYER 2: LSTM (RNN) - 128 units       │  ← RNN LAYER!
│  Remembers previous characters         │
│  return_sequences=True                 │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  LAYER 3: Dropout (0.2)                │
│  Prevents overfitting                  │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  LAYER 4: LSTM (RNN) - 128 units       │  ← ANOTHER RNN LAYER!
│  Further pattern learning              │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  LAYER 5: Dropout (0.2)                │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  LAYER 6: Dense (Fully Connected)      │
│  Softmax activation                    │
│  Outputs probability for each char     │
└────────────────────────────────────────┘
     ↓
OUTPUT: Probability distribution over all characters
        [0.01, 0.05, 0.3, ..., 0.15]  ← Pick 'c' for "me want c"
```

---

## RNN vs Regular Neural Network

### Regular Neural Network (Feed-forward)
```
Input → Hidden → Output
(No memory of previous inputs)
```

### RNN (What we use!)
```
Input₁ → Hidden₁ → Output₁
           ↓ (memory)
Input₂ → Hidden₂ → Output₂
           ↓ (memory)
Input₃ → Hidden₃ → Output₃
```

**RNN remembers previous inputs!** This is perfect for text because:
- "me want c..." → probably "cookie" (not "car")
- "me love m..." → probably "mommy" (not "milk")

---

## Why LSTM Instead of Simple RNN?

### Problem with Simple RNN
```
"me want" → [remember] → [remember] → [forget!] → [lost context]
```
Simple RNNs forget long-term patterns.

### LSTM Solution (What we use)
```
"me want" → [remember] → [remember] → [still remember!] → [use context]
```
LSTM has special "memory cells" that can remember for a long time.

**LSTM Components:**
- **Forget Gate**: What to forget from memory
- **Input Gate**: What new info to add
- **Output Gate**: What to output
- **Cell State**: Long-term memory

---

## Code Breakdown

```python
# This is the RNN part!
model = keras.Sequential([
    layers.Embedding(input_dim=len(chars), output_dim=32, input_length=SEQ_LENGTH),

    layers.LSTM(128, return_sequences=True),  # ← RNN Layer 1
    #           ↑                     ↑
    #     128 neurons         Pass sequences to next layer

    layers.Dropout(0.2),  # Regularization

    layers.LSTM(128),  # ← RNN Layer 2
    #           ↑
    #     128 neurons (only output final state)

    layers.Dropout(0.2),

    layers.Dense(len(chars), activation='softmax')  # Predict next char
])
```

### Parameter Explanation

| Parameter | What it does | Value in our model |
|-----------|-------------|-------------------|
| `LSTM(128)` | Number of memory cells | 128 neurons |
| `return_sequences=True` | Pass all outputs to next layer | First LSTM only |
| `return_sequences=False` | Only output final result | Second LSTM (default) |
| `Dropout(0.2)` | Randomly drop 20% of connections | Prevents overfitting |

---

## How It Learns Patterns

### Training Example

**Input sequence**: `"me want "`
**Target**: `"c"` (from "cookie")

```
Step 1: Feed "me want " into model
        ↓
Step 2: LSTM processes character by character
        'm' → update memory
        'e' → update memory (remembers 'm')
        ' ' → update memory (remembers "me ")
        'w' → update memory (remembers "me w")
        'a' → update memory (remembers "me wa")
        'n' → update memory (remembers "me wan")
        't' → update memory (remembers "me want")
        ' ' → update memory (remembers "me want ")
        ↓
Step 3: Predict next character
        Model outputs: {'c': 0.4, 'o': 0.2, 'm': 0.15, ...}
        ↓
Step 4: Compare with target 'c'
        ↓
Step 5: Adjust weights to improve prediction
```

**After many training examples**, the model learns:
- "me want" → usually followed by objects (cookie, milk, toy)
- "me love" → usually followed by people (mommy, daddy)
- "me see" → usually followed by animals (doggy, kitty)

---

## How to Run (3 Ways)

### Way 1: Direct Execution
```bash
python baby_language_model.py
```

### Way 2: Import and Use
```python
from baby_language_model import BabyLanguageModel, BABY_DATASET

baby = BabyLanguageModel(BABY_DATASET)
baby.build_model()
baby.train(epochs=150)
text = baby.generate_text("me want", length=50)
print(text)
```

### Way 3: Use Helper Script
```bash
python simple_run.py
```

---

## RNN in Action: Step-by-Step Generation

### Generating "me want cookie"

```
Start: "me want "

Step 1:
  Input: "me want " (last 15 chars)
  RNN Memory: [context of "me want "]
  Predict: 'c' (40% probability)
  Output: "me want c"

Step 2:
  Input: "e want c" (last 15 chars)
  RNN Memory: [context of "want c"]
  Predict: 'o' (45% probability)
  Output: "me want co"

Step 3:
  Input: "want co" (last 15 chars)
  RNN Memory: [context of "co" after "want"]
  Predict: 'o' (50% probability)
  Output: "me want coo"

Step 4:
  Input: "want coo" (last 15 chars)
  RNN Memory: [knows it's spelling "coo"]
  Predict: 'k' (60% probability)
  Output: "me want cook"

Step 5:
  Input: "want cook" (last 15 chars)
  RNN Memory: [knows it's "cook"]
  Predict: 'i' (55% probability)
  Output: "me want cooki"

Step 6:
  Input: "ant cooki" (last 15 chars)
  RNN Memory: [completing "cookie"]
  Predict: 'e' (65% probability)
  Output: "me want cookie"
```

The RNN remembers context at each step!

---

## Comparison: RNN vs Transformer

| Feature | RNN (LSTM) | Transformer |
|---------|-----------|-------------|
| Architecture | Sequential processing | Parallel processing |
| Memory | Hidden state | Attention mechanism |
| Speed | Slower (one at a time) | Faster (all at once) |
| Best for | Small datasets | Large datasets |
| Our use case | ✅ Baby model | ⚠️ Overkill |
| Training time | 2-3 minutes | 5-10 minutes |

For a tiny dataset like baby talk, **RNN (LSTM) is perfect!**

---

## Key Takeaways

✅ **Yes, this is an RNN** - Specifically LSTM
✅ **Two LSTM layers** - Stacked for better learning
✅ **Character-level** - Predicts one character at a time
✅ **Sequential memory** - Remembers previous characters
✅ **Perfect for small datasets** - Like our baby talk corpus

---

## Further Reading

- **Understanding LSTM**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **RNN Explained**: https://www.tensorflow.org/guide/keras/rnn
- **Sequence Models**: https://www.coursera.org/learn/nlp-sequence-models

---

**Ready to train your baby-talking RNN? Run the code!** 🚀
