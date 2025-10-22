"""
Baby Language Model - Tiny Transformer Version
Using Hugging Face Transformers with a small GPT-2 style model
Perfect for Google Colab!
"""

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import os

# ============================================================================
# STEP 1: Your baby-talk dataset
# ============================================================================

BABY_DATASET = [
    "me want cookie",
    "me hungry now",
    "mommy me want milk",
    "daddy play with me",
    "me like toy",
    "me no like broccoli",
    "me want go park",
    "me sleepy now",
    "me love mommy",
    "me love daddy",
    "me want more cookie",
    "play ball with me",
    "me see doggy",
    "me pet kitty",
    "me want juice",
    "me no want nap",
    "me big boy now",
    "me big girl now",
    "me do it myself",
    "me help mommy",
    "me help daddy",
    "more milk please",
    "me want up",
    "me go potty",
    "me wash hands",
    "me brush teeth",
    "me wear shoes",
    "me pick this one",
    "me share toy",
    "me say sorry",
    "me good kid",
    "me eat apple",
    "me drink water",
    "me draw picture",
    "me sing song",
    "me read book",
    "me count one two three",
    "me know colors",
    "me see birdie",
    "me hear music",
    "me want hug",
    "me love you",
    "me happy today",
    "me sad now",
    "me tired",
    "me play outside",
    "me ride bike",
    "me swim pool",
    "me build blocks",
    "me paint",
]

# ============================================================================
# STEP 2: Baby Transformer Model Class
# ============================================================================

class BabyTransformer:
    def __init__(self, dataset_texts, model_name="baby-gpt"):
        """
        Initialize a tiny transformer model for baby talk

        Args:
            dataset_texts: List of baby-talk sentences
            model_name: Name for the model
        """
        self.dataset_texts = dataset_texts
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize tokenizer (using GPT2 tokenizer as base)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create a TINY transformer model (very small for fast training)
        self.config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_positions=128,  # Max sequence length
            n_ctx=128,
            n_embd=64,        # Small embedding dimension
            n_layer=4,        # Only 4 transformer layers
            n_head=4,         # 4 attention heads
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )

        print(f"\nTiny Transformer Config:")
        print(f"  - Layers: {self.config.n_layer}")
        print(f"  - Embedding size: {self.config.n_embd}")
        print(f"  - Attention heads: {self.config.n_head}")
        print(f"  - Total parameters: ~{self._count_parameters() / 1e6:.2f}M")

        self.model = None

    def _count_parameters(self):
        """Estimate parameter count"""
        vocab_size = self.config.vocab_size
        n_embd = self.config.n_embd
        n_layer = self.config.n_layer

        # Rough estimate
        embedding_params = vocab_size * n_embd
        transformer_params = n_layer * (12 * n_embd * n_embd)  # Simplified
        return embedding_params + transformer_params

    def prepare_dataset(self, output_file="baby_train.txt"):
        """
        Prepare training data file

        Args:
            output_file: Path to save training data
        """
        # Write dataset to file (required by Hugging Face TextDataset)
        with open(output_file, 'w') as f:
            for text in self.dataset_texts:
                f.write(text + '\n')

        print(f"Dataset saved to {output_file}")
        print(f"Total examples: {len(self.dataset_texts)}")

        self.train_file = output_file
        return output_file

    def train(self, epochs=50, batch_size=4, learning_rate=5e-4):
        """
        Train the baby transformer

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        # Prepare dataset
        self.prepare_dataset()

        # Create model
        self.model = GPT2LMHeadModel(self.config)
        self.model.to(self.device)

        print(f"\nModel has {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Prepare dataset for training
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.train_file,
            block_size=64  # Sequence length
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./baby_model_output",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            logging_dir='./baby_logs',
            report_to=[],  # Disable wandb
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # Train!
        print("\nStarting training...")
        trainer.train()

        print("\nTraining complete!")

    def generate(self, prompt="me want", max_length=30, temperature=1.0, num_return_sequences=1):
        """
        Generate baby talk!

        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            temperature: Randomness (0.5=conservative, 1.0=balanced, 1.5=creative)
            num_return_sequences: Number of different outputs to generate
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")

        self.model.eval()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and return
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def save_model(self, save_dir="baby_transformer_model"):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")

        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

    def load_model(self, save_dir="baby_transformer_model"):
        """Load a trained model"""
        self.model = GPT2LMHeadModel.from_pretrained(save_dir)
        self.model.to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded from {save_dir}")


# ============================================================================
# STEP 3: Main training script
# ============================================================================

def main():
    print("=" * 60)
    print("Baby Transformer Language Model")
    print("=" * 60)

    # Create model
    baby_transformer = BabyTransformer(BABY_DATASET)

    # Train (this will take a few minutes on CPU, faster on GPU)
    baby_transformer.train(epochs=100, batch_size=4, learning_rate=5e-4)

    # Generate baby talk!
    print("\n" + "=" * 60)
    print("Baby Says:")
    print("=" * 60)

    prompts = ["me want", "me love", "me see", "me play", "daddy"]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)

        # Generate with different temperatures
        for temp in [0.7, 1.0, 1.3]:
            outputs = baby_transformer.generate(
                prompt=prompt,
                max_length=20,
                temperature=temp,
                num_return_sequences=1
            )
            print(f"[temp={temp}] {outputs[0]}")

    # Save model
    baby_transformer.save_model()

    return baby_transformer


# ============================================================================
# STEP 4: Quick generation after training
# ============================================================================

def quick_generate(prompt="me want", num_outputs=5):
    """Generate baby talk using saved model"""
    baby = BabyTransformer(BABY_DATASET)
    baby.load_model()

    print(f"Baby says (prompt: '{prompt}'):\n")

    outputs = baby.generate(
        prompt=prompt,
        max_length=25,
        temperature=1.0,
        num_return_sequences=num_outputs
    )

    for i, text in enumerate(outputs, 1):
        print(f"{i}. {text}")


if __name__ == "__main__":
    # Check if transformers is installed
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Please install transformers: pip install transformers")
        exit(1)

    # Train the model
    baby_transformer = main()

    # Generate more examples
    print("\n\n" + "=" * 60)
    print("More Baby Talk Examples")
    print("=" * 60)
    quick_generate("me want", num_outputs=5)
