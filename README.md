# AI_gen_books
用ai和prompt 生成的书，测试。


## bencharmk on generation.
testing on ollama 0.31, docker aio install, intel 13700kf ddr4 64GB.

asked question: tell me how pcb is done, and how long can it keep?
how to get result, load the model, and make a new chat, ask this question.

response token per second:
| Test Item                     | Tesla P8 8GB  | 2080 Ti 22GB 250W | RTX3090 24GB 350W  | 4090 Ti 24GB 450W |
|-------------------------------|---------------|-------------------|--------------------|-------------------|
| Mistral-Nemo  12.2B           |               |     60.05            |                    |               | 
| llama3.1 8B                   |               |      85.54            |                |               |
| qwen2 7.6B                    |               |        90.57       |             |               |




