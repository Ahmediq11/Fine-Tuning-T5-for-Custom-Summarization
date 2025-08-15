# Fine-Tuning T5 for Dialogue Summarization 💬

This repository contains a complete, end-to-end project for fine-tuning a T5 (Text-to-Text Transfer Transformer) model on the `samsum` dataset to summarize dialogues and conversations. The entire workflow is built using the Hugging Face ecosystem, demonstrating advanced techniques for training large sequence-to-sequence models efficiently.

The goal is to take a multi-turn conversation as input and generate a concise, human-readable summary, a task at which the T5 architecture excels.

## ✨ Key Features

  * **State-of-the-Art Model**: Utilizes `t5-large`, a powerful pre-trained model from Google, known for its outstanding performance on sequence-to-sequence tasks.
  * **Efficient Training Techniques**: Implements several strategies to train a large model on a standard GPU:
      * **Gradient Accumulation**: Simulates a very large batch size (200) to stabilize training without high VRAM usage.
      * **Mixed-Precision Training (`fp16`)**: Reduces memory footprint and significantly speeds up the training process on modern GPUs.
  * **Custom Dataset Fine-Tuning**: Demonstrates the full process of adapting a general-purpose model to a specific domain (dialogue summarization) using the `samsum` dataset.
  * **End-to-End Workflow**: Covers every step from data loading and cleaning to tokenization, training, and inference.
  * **Hugging Face Integration**: Leverages the high-level `Trainer` API for a streamlined workflow and the `pipeline` function for easy, production-ready inference.

-----

## ⚙️ Project Workflow

The project follows a structured machine learning pipeline:

1.  **Environment Setup**: Installs all necessary libraries, including `transformers`, `datasets`, and `sentencepiece` (a key dependency for the T5 tokenizer).
2.  **Data Loading and EDA**: The `samsum` dataset, containing dialogues and their summaries, is loaded. An exploratory data analysis (EDA) is performed to visualize the length distribution of dialogues and summaries, confirming their suitability for the summarization task.
3.  **Model and Tokenizer Loading**: The `t5-large` model and its corresponding tokenizer are loaded from the Hugging Face Hub.
4.  **Data Preprocessing**:
      * The dataset is cleaned by filtering out any examples with missing values.
      * A tokenization function is created to convert both the input dialogues and target summaries into numerical IDs that the model can understand. This is applied to the entire dataset using the `.map()` method.
5.  **Trainer Configuration**:
      * A `DataCollatorForSeq2Seq` is used to intelligently pad batches of tokenized inputs and labels independently for maximum efficiency.
      * `TrainingArguments` are configured with hyperparameters, including gradient accumulation and mixed-precision training, to manage memory and speed up the process.
      * The `Trainer` is instantiated, bringing together the model, tokenizer, datasets, and training configurations.
6.  **Training and Saving**: The model is fine-tuned on the `samsum` dataset by calling `trainer.train()`. After training, the final model artifacts are saved to a directory for later use.
7.  **Inference**: The fine-tuned model is loaded into a `summarization` `pipeline` to easily generate summaries for new, unseen dialogues.

-----

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-enabled GPU to take full advantage of the training optimizations.

### Installation

1.  Clone the repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  Install the required Python libraries:

    ```bash
    pip install -U transformers datasets accelerate sentencepiece py7zr
    ```

-----

## ▶️ How to Use the Trained Model

The fine-tuned model is saved in the `t5_samsum_summarization` directory. The easiest way to use it for prediction is with the Hugging Face `pipeline` function, which handles all the necessary preprocessing and post-processing steps.

```python
from transformers import pipeline
import torch

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Load the saved model into a summarization pipeline
summarizer_pipe = pipeline('summarization', model='t5_samsum_summarization', device=device)

# A new, custom dialogue to summarize
custom_dialogue="""
Laxmi Kant: what work you planning to give Tom?
Juli: i was hoping to send him on a business trip first.
Laxmi Kant: cool. is there any suitable work for him?
Juli: he did excellent in last quarter. i will assign new project, once he is back.
"""

# Get the summary
output = summarizer_pipe(custom_dialogue)

# Print the result
print(output[0]['summary_text'])
```

### Expected Output

```
Juli is planning to send Tom on a business trip. He will assign a new project to him once he is back from the trip.
```

-----

## 📚 Dataset

This project uses the **`samsum`** dataset, which is a collection of about 16k messenger-like conversations with summaries. This data is specifically created for the task of dialogue summarization. It was loaded from the Hugging Face Hub using the `datasets` library.

-----

## 📄 License

This project is licensed under the MIT License. See the `MIT` file for more details.
