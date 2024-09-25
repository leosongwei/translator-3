# 翻译器3号

字幕翻译一直是一个比较tricky的事情，你不太好囫囵地把字幕交给商业模型翻译，因为它会弄乱位置，也不太好一句一句地交给商业模型，因为没有上下文，同一个人物名字或者地名可能变来变去让你摸不着头脑。

基本想法和功能：

1. 利用P-tuning V2编码上下文，使得模型口径连贯。
   * 上下文可以是一个一部动画所有的字幕也可以是一集。
   * 如果已知人名的对应翻译，也可以加入上下文中。
   * 手动实现了ptuning，因为peft不能让P-tuning和LoRA一起使用。
   * [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)
2. 利用LoRA微调使得预训练模型能够利用上下文embedding进行翻译。
3. prompt中带有相邻的几句话提供局部上下文信息（当然也会显著增加序列长度消耗显存）。
4. LoRA训练时采用Instruction Masking，防止上下文信息起反作用。
   * Mask掉prompt部分，仅对答案部分计算loss。 
   * LoRA训练的时候发现加入上下文过后性能反而下降，生成的语句比较混乱，才意识到有这一层。
   * [LLM Research Insights: Instruction Masking and New LoRA Finetuning Experiments](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)
5. 适配一个比较新的模型，具有比较低的推理开销，最好也能具有比较低的训练开销。
   * 一开始在ChatGLM3上取得了比较好的结果，现在适配了Qwen2.5-3B

局限：

1. 我不好给你我的训练数据，只能给你一个微调结果。
2. 一致性只能达到一个我堪堪满意的程度（比我上一个自用的翻译器好一些，但没有好很多），且没时间充分测试，具体性能尚不明晰。
   - BLEU测试在ChatGLM3-6B上涨点，在Qwen2.5-3B上反而下降，推测是训练不充分导致模型感到费解。
   - 具体效果参见文件`translate_result_example.jsonl`，翻译了《天空之城》的字幕，pred字段是翻译结果。
3. Qwen2.5-3B的微调内存消耗迷之高，本以为只需要小于10GB，结果几乎消耗20GB，看来需要我进一步debug。

## 用法

1. 训练大量P-tunings
   - 将你的数据集组织成jsonl文件形式，每一个文件夹里可以包含多个jsonl（注意保持顺序）
     - 每一行格式为`{"jp": "日文部分", "cn": "对应的中文部分"}`
   - 把所有的文件夹的位置放到`data_registry.json`中。（该文件对应我自己的设置，你需要修改该文件）
   - 执行`ptuning_trainer.py`
   - 会产生文件夹`ptuning_results`。
2. 训练LoRA
   - 现在我们有了大量的ptunings，可以训练LoRA了。
   - 执行`lora_trainer.py`
   - 将会得到`lora_result`文件夹
3. 使用：执行run.py，有说明。 
   - 记得先在venv中安装`requirements.txt`中的内容