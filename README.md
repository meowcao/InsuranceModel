# 保险知识问答助手

## 介绍

​	欢迎使用我们的保险知识问答大语言模型！这个模型是基于`ChineseNlpCorpus`提供的丰富保险领域数据集开发而成。我们的数据集涵盖了用户的各种保险相关提问、网友的回答以及最佳回答，旨在为您提供全面、准确的保险知识解答。

​	我们的模型是您在保险领域的智囊团，具备丰富的专业知识和应用能力。无论您是想了解保险的基本概念、不同类型的保险，还是需要指导理赔流程或选择合适的保险政策，我们都能为您提供帮助。

具体如何实现全流程的 chat-AI 微调，可参考本仓库-QQQQQQ

如何学习大模型部署和微调请参考：[开源大模型食用指南](https://github.com/datawhalechina/self-llm.git) 以及 [书生·浦语大模型实战营课程](https://github.com/InternLM/tutorial.git)

![structure](imgs/structure.png)

## OpenXlab 模型

保险知识问答助手使用的是InternLM 的 7B 模型，模型参数量为 7B，模型已上传，可以直接下载推理。

| 基座模型         | 微调数据量          | 训练次数 | 下载地址 |
| ---------------- | ------------------- | -------- | -------- |
| InternLM-chat-7b | 46933 conversations | 5 epochs |          |

## 数据集

​	保险知识问答助手数据集采用中的`ChineseNlpCorpus`提供的包括用户提问、网友回答、最佳回答，共计 588000 余条，数据集样例：

```
"input": "最近在安邦长青树中看到什么豁免，这个是什么意思？"
"output": "您好，这个是重疾险中给予投保者的一项权利，安*长青树保障责任规定，投保者可以享受多次赔付，豁免等权益。也就是说不同轻症累计5次赔付，理赔1次轻症豁免后期所交保费，人性化的设计，无需加保费。"
"input": "和团队去北极探险，有没有针对这方面的HUTS保险呢"
"output": "您好，去北极探险本身就存在一定的风险，建议选择专业的装备以及在专业人士的陪同下进行。至于保险，市面上关于此类的保险并不多，不过HUTS保险中却有一款专门针对南北极旅游的定制产品，保障内容充足，户外伤害、医疗保障甚至的紧急救援都具备，详情可以多了解下。"
```

### 数据处理与整理

1. 数据集是以CSV格式存储的，第一行为列名，分别为 `title`, `reply`, `is_best`。
2. 数据集把数据分类为优质回答（is_best=1）与劣质回答（is_best=0），需要过滤掉 `is_best` 列为0的数据。

使用如下脚本文件



```python
import csv
import json

def convert_csv_to_json(input_file, output_file):
    conversations = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if int(row['is_best']) == 1:  # 只处理 is_best 为 1 的数据
                conversation = {
                    "system": "当你向我询问有关保险的问题时，请放心，我是一位专业的保险知识问答专家。我可以帮助您澄清保险的基本概念和作用，解释不同类型保险的特点，以及回答关于保险合同、理赔流程或保险政策的任何问题。您可以随时向我提出关于健康、车辆、房屋和人寿保险等方面的疑问。如果您想了解如何提出保险理赔申请或需要关于保险理赔流程的详细信息，请告诉我，我将为您提供专业的指导和建议。",
                    "input": row['title'],
                    "output": row['reply']
                }
                conversations.append({"conversation": [conversation]})
    
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(conversations, jsonfile, ensure_ascii=False, indent=4)

# 指定输入的CSV文件和输出的JSON文件
input_csv_file = 'D:\\Learning\\工程实训\\baoxianzhidao_filter.csv'
output_json_file = 'D:\\Learning\\工程实训\\output.json'

# 调用函数进行转换
convert_csv_to_json(input_csv_file, output_json_file)

print(f"转换完成，结果已保存到 {output_json_file}")

```

- `convert_csv_to_json` 函数负责读取CSV文件，过滤和转换数据，并将结果写入JSON文件。
- 我们使用 `csv.DictReader` 来读取CSV文件，并以字典形式访问每一行的数据。
- 对于每一行数据，根据 `is_best` 的值决定是否添加到最终的JSON输出中。
- 最后，使用 `json.dump` 将转换后的数据写入到指定的JSON文件中。

## 微调

  使用 `XTuner `训练， `XTuner `有各个模型的一键训练脚本，很方便。且对` InternLM2 `的支持度最高。

### XTuner

  使用 `XTuner` 进行微调，具体脚本可参考`configs`文件夹下的脚本，脚本内有较为详细的注释。

| 基座模型         | 配置文件                               |
| ---------------- | -------------------------------------- |
| internlm-chat-7b | internlm_chat_7b_qlora_medqa2019_e3.py |

微调方法如下:

1. 根据基座模型复制上面的配置文件，将模型地址`pretrained_model_name_or_path`和数据集地址`data_path`修改成自己的

```
conda activate xtuner0.1.9
cd ~/ft-medqa
xtuner train  internlm_chat_7b_qlora_medqa2019_e3.py --deepspeed deepspeed_zero2
```

2. 将得到的 PTH 模型转换为 HuggingFace 模型

```
internlm_chat_7b_qlora_medqa2019_e3
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_medqa2019_e3.py ./work_dirs/internlm_chat_7b_qlora_medqa2019_e3/epoch_1.pth ./hf
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
```

### Chat

```
xtuner chat ./merged --prompt-template internlm_chat
```

## 本地网页部署

```
streamlit run /root/ft-medqa/code/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

效果演示

![local_2](imgs/local_2.png)

![local_1](imgs/local_1.png)

