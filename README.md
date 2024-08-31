# AI_gen_books
用ai和prompt 生成的书，测试。


## Benchmark on Model Generation

This benchmark was conducted using Ollama 0.31 with the Docker AIO installation. The tests were run on the following hardware configurations:

- **Tesla P8** on Xeon 2697v3
- **RTX 2080 Ti** on Intel 13700KF with 64GB DDR4 RAM
- **RTX 3090** on Intel 13700KF with 64GB DDR4 RAM
- **RTX 4090** on Intel 14900KS

### Test Procedure

**Question Asked**: *"Tell me how PCB is made, and how long can it keep?"*

**Methodology**: 
1. Load the respective model on each hardware setup.
2. Start a new chat session.
3. Ask the question.
4. Measure the response tokens per second.

### Results

| Model                         | Tesla P8 8GB 70W  | RTX 2080 Ti 22GB 250W | RTX 3090 24GB 350W | DUAL 3090 24GB  | DUAL 3090 + 2080 Ti | 3090 + 2080 Ti | RTX 4090 24GB 450W |
|-------------------------------|-------------------|----------------------|-------------------|-----------------|---------------------|----------------|----------------------|
| **Mistral-Nemo 12.2B**         | 9.59  (partial)   | 60.05                | 82.26             |                 |                     |                |     95.75           |
| **LLaMA 3.1 8B**               | 24.85             | 85.54                | 115.73            |                 |                     |                |     131.65          |
| **Qwen2 7.6B**                 | 25.33             | 90.57                | 121.84            |                 |                     |                |                     |
| **LLaMA 3.1 70B**              |                   |                      |                   | 18.45           | 16.06               | 15.29          |                     |

### GPU Specs

| Specification          | Tesla P8 | RTX 2080 Ti | RTX 3090 | RTX 4090 |
|------------------------|----------|-------------|----------|----------|
| **CUDA Cores**         | 2,048    | 4,352       | 10,496   | 16,384   |
| **Memory Bandwidth**   | 224 GB/s | 616 GB/s    | 936 GB/s | 1,008 GB/s |
| **VRAM Size**          | 8 GB     | 22 GB       | 24 GB    | 24 GB    |
| **Price 2024Q3 (USD)** | ~$80     | ~$350     | ~$750    | ~$2000   |
| **TOKEN/100USD**       | 31.1       | 24.4      | 15.4     |          |
