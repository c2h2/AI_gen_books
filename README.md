# AI_gen_books
用ai和prompt 生成的书，测试。


## Benchmark on Model Generation

This benchmark was conducted using Ollama 0.31 with the Docker AIO installation. The tests were run on the following hardware configurations:

- **Tesla P8** on Xeon 2697v3
- **2080 Ti** on Intel 13700KF with 64GB DDR4 RAM
- **RTX 3090** on Intel 13700KF with 64GB DDR4 RAM
- **4090 Ti** on Intel 14900KS

### Test Procedure

**Question Asked**: *"Tell me how PCB is done, and how long can it keep?"*

**Methodology**: 
1. Load the respective model on each hardware setup.
2. Start a new chat session.
3. Ask the question.
4. Measure the response tokens per second.

### Results

| Model                         | Tesla P8 8GB 70W  | 2080 Ti 22GB 250W | RTX 3090 24GB 350W | 4090 Ti 24GB 450W |
|-------------------------------|-------------------|-------------------|-------------------|-------------------|
| Mistral-Nemo 12.2B             |  9.59  (partial)   |      60.05        |                   |                   |
| LLaMA 3.1 8B                   |      24.85        |      85.54        |                   |                   |
| Qwen2 7.6B                     |      25.33        |      90.57        |                   |                   |
