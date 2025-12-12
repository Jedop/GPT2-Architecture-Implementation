# GPT2-Architecture-Implementation

An implementation of the GPT-2 architecture in PyTorch, built from scratch. This project features a custom weight-loading pipeline that maps and transposes official OpenAI pre-trained weights (via Hugging Face) into a custom model definition, enabling inference with GPT2 Small Parameters.

## Key Features

*   **Architecture from Scratch:** Implemented the full Transformer decoder stack, including `MultiHeadAttention` (with causal masking), `FeedForward` networks, and `LayerNorm`.
*   **Industrial Attention Implementation:** Used the "Single Matrix" QKV projection method (optimizing matrix multiplications) rather than split linear layers, matching standard production implementations.
*   **Weight Surgery:** Built a custom loading script to map Hugging Face's `Conv1D` weights to PyTorch `Linear` layers, handling critical tensor transpositions and shape mismatches programmatically.
*   **Interactive Inference:** Includes a CLI loop for real-time text generation using the ported weights.

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/Jedop/GPT2-Architecture-Implementation.git
   cd GPT2-Architecture-Implementation
   ```
2. Install the required dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   E.g.
   ```sh
   python Transformer_Architecture.py
   ```
## Example Output

```text
Loading weights from Hugging Face GPT-2...
Weights loaded successfully.

==================================================
GPT-2 Inference Engine Ready
==================================================

Prompt (Type 'exit' or 'quit' to exit.) > The scientist discovered a new              

Response:
The scientist discovered a new way how galaxies will move. In a very strange approach, he mapped how galaxies will move relative to one another, which is why every galaxy (as measured by "ratio" - we assume first) will drive past their sun. In working with

--------------------------------------------------
```

## Tech Stack
*   **Python 3.12**
*   **PyTorch (CUDA supported)**
*   **Hugging Face Transformers** (For Tokenizer and Source Weights)

## References
*   Based on concepts from Andrej Karpathy's `nanoGPT`.
