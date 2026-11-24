
# Lab 03 – Developing a Tiny Transformer Language Model

## 1. Laboratory Session

**Lab:** 03 – Developing a Tiny Transformer Language Model  
**Module:** PUSL3205 – Large Language Models and Generative AI 
**Development Environment:** Python, PyTorch, Hugging Face Datasets, Jupyter Notebook, Visual Studio Code

## 2. Objective

The objective of this laboratory session was to develop a small Transformer-based language model using the TinyStories dataset.

The model was designed to predict the next character in a text sequence and generate stories from a given starting prompt.

This practical covered dataset exploration, character-level tokenization, sequence preparation, Transformer architecture development, model training, evaluation, text generation, experimentation, and model checkpoint saving.

## 3. Tasks and Implementation

### Task 1 – Load and Explore the Dataset

Loaded a subset of 1,000 stories from the `roneneldan/TinyStories` dataset using Hugging Face Datasets.

Explored sample stories and analyzed their lengths to understand the dataset structure.

**Evidence:**

- [Dataset Loading](screenshots/Task01_Dataset.png)
- [Sample Stories](screenshots/Task01_Stories.png)
- [Story Length Analysis](screenshots/Task01_Story_Lengths.png)

### Task 2 – Tokenize the Dataset

Divided the selected dataset into 900 training stories and 100 validation stories.

Implemented character-level tokenization by creating a vocabulary of unique characters.

Developed encoding and decoding functions to convert text into numerical token IDs and reconstruct text from those IDs.

**Evidence:**

- [Dataset Split](screenshots/Task02_Dataset_Split.png)
- [Character Tokenization](screenshots/Task02_Character_Tokenization.png)
- [Encoding and Decoding](screenshots/Task02_Encoding_Decoding.png)

### Task 3 – Prepare Training Data

Prepared input and target sequences for next-character prediction.

Used a sequence length of 64 characters and a batch size of 32.

Each target sequence was shifted one character ahead of its input sequence, allowing the model to learn to predict the next character.

**Evidence:**

- [Input and Target Shapes](screenshots/Task03_Input_Target_Shapes.png)
- [Sequence Example](screenshots/Task03_Sequence_Example.png)

### Task 4 – Build the Transformer Architecture

Developed a small Transformer-based language model using PyTorch.

The architecture included:

- Token embeddings
- Positional embeddings
- Masked multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- A linear output layer for next-character prediction

The original Transformer used two Transformer layers, two attention heads, and an embedding dimension of 64.

Causal masking was implemented to prevent the model from attending to future characters during prediction.

**Evidence:**

- [Transformer Architecture – Part 1](screenshots/Task04_Architecture_01.png)
- [Transformer Architecture – Part 2](screenshots/Task04_Architecture_02.png)

### Task 5 – Initialize the Model

Initialized the Transformer model and configured the training components.

Used the cross-entropy loss function to measure next-character prediction errors.

Configured the AdamW optimizer with a learning rate of 0.001 to update model parameters during training.

**Evidence:**

- [Model Initialization](screenshots/Task05_Model_Initialization.png)
- [Loss Function and Optimizer](screenshots/Task05_Loss_Optimizer.png)

### Task 6 – Train the Transformer Model

Trained the original Transformer model for 15 epochs using 100 training steps per epoch.

The training process included forward propagation, cross-entropy loss calculation, backpropagation, and optimizer updates.

Training and validation losses were recorded to monitor learning progress.

**Training Results:**

| Metric | Result |
|---|---:|
| Training Epochs | 15 |
| Steps per Epoch | 100 |
| Final Training Loss | 1.5671 |
| Final Validation Loss | 1.5457 |

Both training and validation losses decreased throughout the training process, indicating that the model learned character-sequence patterns from the dataset.

**Evidence:**

- [Evaluation Function](screenshots/Task06_Evaluation_Function.png)
- [Training Code](screenshots/Task06_Training_Code.png)
- [Training Results](screenshots/Task06_Training_Results.png)
- [Final Loss](screenshots/Task06_Final_Loss.png)

### Task 7 – Monitor Training

Visualized the recorded training and validation losses using Matplotlib.

The learning curves showed a decreasing trend across the 15 epochs.

This indicated improvements in the model's next-character prediction performance during training.

**Evidence:**

- [Training Loss Curve Screenshot](screenshots/Task07_Training_Loss_Curve.png)
- [Saved Training Loss Graph](results/Training_Loss_Curve.png)

### Task 8 – Evaluate the Model

Evaluated the trained Transformer using validation loss and character-level perplexity.

Validation loss was calculated using cross-entropy loss on sampled validation batches.

Perplexity was calculated by exponentiating the validation loss.

**Evaluation Results:**

| Metric | Result |
|---|---:|
| Validation Loss | 1.5776 |
| Validation Perplexity | 4.84 |

The evaluation results indicate that the model learned useful patterns for predicting characters in unseen text.

Because character-level tokenization was used, the perplexity value represents character-level prediction uncertainty.

**Evidence:**

- [Model Evaluation](screenshots/Task08_Model_Evaluation.png)

### Task 9 – Generate Stories

Implemented autoregressive text generation using the trained Transformer.

The model generated text one character at a time by sampling from predicted character probabilities.

Used the starting prompt:

`Once upon a time`

The model produced text containing recognizable words and sentence-like patterns, although grammatical errors, repetition, and incomplete words were present.

These limitations were expected due to the small model architecture and limited training dataset.

**Evidence:**

- [Generation Function](screenshots/Task09_Generation_Function.png)
- [Generated Story 1](screenshots/Task09_Generated_Story_01.png)
- [Generated Story 2](screenshots/Task09_Generated_Story_02.png)
- [Generated Stories Text File](results/Generated_Stories.txt)

### Task 10 – Experiment with Model Settings

Conducted an experiment to investigate the effect of changing the number of attention heads.

Compared the original Transformer with two attention heads against a second Transformer using four attention heads.

Both models used two Transformer layers, an embedding dimension of 64, and the same training schedule.

The models were evaluated using the same sampled validation batches for comparison.

**Comparison Results:**

| Configuration | Attention Heads | Validation Loss | Perplexity |
|---|---:|---:|---:|
| Original Transformer | 2 | 1.5855 | 4.88 |
| Comparison Transformer | 4 | 1.6387 | 5.15 |

**Observations:**

The original Transformer with two attention heads achieved a slightly lower validation loss of 1.5855 compared with 1.6387 for the four-head comparison model.

The original model also achieved lower character-level perplexity.

Both models generated text from the same starting prompt, but their outputs contained grammatical mistakes, incomplete words, and inconsistent sentence structures.

The results suggest that increasing the number of attention heads did not improve validation performance in this experiment.

However, additional experiments would be required to determine whether this difference is consistent across repeated training runs.

**Evidence:**

- [Comparison Training](screenshots/Task10_Comparison_Training.png)
- [Comparison Training Epochs](screenshots/Task10_Comparison_Epochs.png)
- [Validation Comparison](screenshots/Task10_Validation_Comparison.png)
- [Generated Stories Comparison](screenshots/Task10_Generated_Stories.png)
- [Final Comparison Results](screenshots/Task10_Final_Results.png)

### Task 11 – Save the Trained Model

Saved the trained Transformer models as PyTorch checkpoint files.

Each checkpoint contains the trained model parameters, character vocabulary, vocabulary size, and model configuration.

Both the original and comparison models were saved separately for future use.

**Saved Models:**

- `models/Tiny_Transformer_Original.pt`
- `models/Tiny_Transformer_Comparison.pt`

**Evidence:**

- [Original Model Checkpoint](screenshots/Task11_Original_Model.png)
- [Comparison Model Checkpoint](screenshots/Task11_Comparison_Model.png)
- [Saved Model Verification](screenshots/Task11_Saved_Models.png)

## 4. Results and Observations

The original Transformer successfully completed 15 epochs of training using a subset of the TinyStories dataset.

The final training loss was **1.5671**, while the final validation loss recorded during training was **1.5457**.

In Task 8, the model achieved a validation loss of **1.5776** and a character-level perplexity of **4.84**.

The training and validation loss curves generally decreased, showing that the model improved its character prediction capabilities during training.

In Task 10, the original two-head Transformer achieved a validation loss of **1.5855**, compared with **1.6387** for the four-head model when evaluated using the same sampled validation batches.

The original configuration performed slightly better in this experiment.

Both Transformer models generated text from the prompt "Once upon a time". The outputs contained recognizable words and some sentence-like patterns, but also showed grammatical errors, repetition, and incomplete words.

The different validation loss values recorded in Tasks 6, 8, and 10 resulted from evaluations using different randomly sampled validation batches.

Overall, the results demonstrated the capabilities and limitations of training a small character-level Transformer language model on a limited dataset.

## 5. Mechanisms for Improving Model Performance

Several mechanisms were implemented or investigated during this laboratory session.

- **Causal Masking:** Prevented the model from attending to future characters during next-character prediction.
- **Multi-Head Attention:** Allowed the Transformer to learn relationships between different character positions.
- **Positional Embeddings:** Provided information about the positions of characters within the input sequence.
- **Residual Connections:** Supported information flow across Transformer layers.
- **Layer Normalization:** Helped support stable optimization during training.
- **AdamW Optimizer:** Updated the model parameters to reduce prediction errors.
- **Validation Monitoring:** Allowed training progress and performance on unseen sequences to be observed.
- **Attention-Head Experimentation:** Investigated how changing the number of attention heads affected validation loss and generated text.

Possible future improvements include using a larger training dataset, experimenting with longer training schedules, and applying subword tokenization.

These are potential extensions and were not implemented as part of this laboratory session.

## 6. Learning Outcomes

Through this laboratory session, I gained practical experience in:

1. Loading and exploring text datasets using Hugging Face.
2. Implementing character-level tokenization.
3. Preparing input and target sequences for next-character prediction.
4. Understanding and implementing Transformer architecture components.
5. Training a language model using PyTorch.
6. Monitoring training and validation losses.
7. Evaluating a model using cross-entropy loss and perplexity.
8. Generating text autoregressively from a starting prompt.
9. Experimenting with different attention-head configurations.
10. Comparing model performance using validation loss and generated text.
11. Saving trained model parameters and configurations for future use.

## 7. Project Files

- **Jupyter Notebook:** [Tiny_Transformer.ipynb](notebooks/Tiny_Transformer.ipynb)
- **Reference Code:** [Reference Files](reference/)
- **Trained Models:** [Models](models/)
- **Generated Results:** [Results](results/)
- **Screenshots:** [Screenshots](screenshots/)

## 8. Conclusion

This laboratory session provided hands-on experience in developing a small Transformer-based language model using PyTorch and the TinyStories dataset.

The practical demonstrated the complete workflow from dataset exploration and tokenization to Transformer architecture development, model training, evaluation, text generation, experimentation, and checkpoint saving.

The experiment also showed that increasing the number of attention heads did not necessarily improve validation performance under the tested conditions.

Overall, the laboratory strengthened my understanding of Transformer-based language models and the factors that influence their training and text-generation performance. 
