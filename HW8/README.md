# 弱智吧数据微调数据构造实验
## 环境配置
```shell
studio-conda xtuner0.1.17

# 激活环境
conda activate xtuner0.1.17
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir -p /root/ruozhi && cd /root/ruozhi

# 拉取 0.1.17 的版本源码
git clone -b v0.1.17  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.15 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd /root/xtuner0117/xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```
## 上传数据
下载[数据](https://opendatalab.org.cn/OpenDataLab/COIG-CQIA)到本地后，上传至线上机器
![](1.png)

## 数据处理 
原始数据格式
```json
{
    "instruction": "天下没有不散的筵席，那么我们相聚的意义又是什么",
    "input": "",
    "output": "尽管我们相聚的时光有限，但是相聚的意义在于创造美好的回忆和珍贵的关系。相聚让我们感受到彼此的关怀、支持和友情。我们可以一起分享喜悦、快乐和困难，互相支持和激励。相聚也可以是一个机会，让我们相互了解、学习和成长。最重要的是，相聚能够带给我们真实的人际交往和情感交流，让我们感受到生活的丰富和美好。所以，即使相聚的时间有限，我们仍然要珍惜这份意义和与彼此相聚的时光。",
    "task_type": {
        "major": [
            "问答"
        ],
        "minor": [
            "逻辑问答",
            "隐喻理解"
        ]
    },
    "domain": [
        "通用"
    ],
    "metadata": "暂无元数据信息",
    "answer_from": "llm",
    "human_verified": true,
    "copyright": "暂无版权及作者信息"
}
```
目标格式
```json
{
    "messages": [
        {
            "role": "user",
            "content": ""
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]
}
```
在 `/root/ruozhi/data` 目录中进行数据处理。`touch /root/ft/data/generate_data.py` 创建 `generate_data.py` 文件:
```py
import json

# 原始和目标文件路径
input_file_path = '/root/ruozhi/ruozhiba.json'
output_file_path = '/root/ruozhi/ruozhiba_format.json'

# 打开原始文件进行读取
with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # 将每行的json字符串转换为字典
        original_data = json.loads(line)

        # 转换格式
        converted_data = {
            "messages": [
                {
                    "role": "user",
                    "content": original_data["instruction"]
                },
                {
                    "role": "assistant",
                    "content": original_data["output"]
                }
            ]
        }

        # 写入新的jsonl文件，确保中文不被转义
        json.dump(converted_data, output_file, ensure_ascii=False)
        output_file.write('\n')  # 每行后添加换行符，以保持jsonl格式

```
修改完成后运行 generate_data.py 文件即可。
```shell
# 运行代码
python /root/ruozhi/data/generate_data.py
```

## 划分训练集和测试集
在 `/root/ruozhi/data` 目录中进行数据处理。`touch /root/ft/data/split_data.py` 创建 `split_data.py` 文件:
```py
import json

# 输入文件路径和输出文件路径
input_file_path = '/root/ruozhi/ruozhiba_format.json'
train_file_path = '/root/ruozhi/train.json'
test_file_path = '/root/ruozhi/test.json'

# 读取原始文件，将所有行存储在一个列表中
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 计算训练集和测试集的分割点
split_index = int(len(lines) * 0.7)

# 拆分数据
train_lines = lines[:split_index]
test_lines = lines[split_index:]

# 写入训练集
with open(train_file_path, 'w', encoding='utf-8') as file:
    for line in train_lines:
        file.write(line)

# 写入测试集
with open(test_file_path, 'w', encoding='utf-8') as file:
    for line in test_lines:
        file.write(line)
```

## 开始自定义微调
复制internlm2-chat-1.8B模型
```shell
mkdir -p /root/ruozhi/model
cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/ruozhi/model/
```
### 准备配置文件
本案例基于internlm2_1_8b_qlora_alpaca_e3.py进行修改
```shell
# 创建一个存放 config 文件的文件夹
mkdir -p /root/ruozhi/config

# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ruozhi/config/
```

重命名配置文件为'internlm2_1_8b_qlora_ruozhiba_e3.py'，并修改配置文件：
```diff
# 修改模型地址（在第27行的位置）
- pretrained_model_name_or_path = 'internlm/internlm2-1_8b'
+ pretrained_model_name_or_path = '/root/ruozhi/model'

# 修改数据集地址为本地的json文件地址（在第31行的位置）
- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = '/root/ruozhi/data/train.jsonl'

# 修改评估的问题（在第59到61行的位置）
- evaluation_inputs = ['请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai']
+ evaluation_inputs = ['为什么我爸妈结婚的时候没邀请我参加婚礼']

# 把 OpenAI 格式的 map_fn 载入进来（在第15行的位置）
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory

# 将原本是 alpaca 的地址改为是 json 文件的地址（在第102行的位置）
- dataset=dict(type=load_dataset, path=alpaca_en_path),
+ dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),

# 将 dataset_map_fn 改为通用的 OpenAI 数据集格式（在第105行的位置）
- dataset_map_fn=alpaca_map_fn,
+ dataset_map_fn=openai_map_fn,
```

### 启动训练
```shell
xtuner train /root/ruozhi/config/internlm2_1_8b_qlora_ruozhiba_e3.py --work-dir /root/ruozhi/train --deepspeed deepspeed_zero2
```
训练开始：
![](2.png)

不同轮次：
![](3.png)
![](4.png)
![](5.png)
![](6.png)
![](7.png)
![](8.png)
![](9.png)
![](10.png)
![](11.png)
![](12.png)

### pth 转 huggingface
```shell
mkdir -p /root/ft-ruozhiba/huggingface
xtuner convert pth_to_hf /root/ruozhi/train/internlm2_1_8b_qlora_ruozhiba_e3.py /root/ruozhi/train/iter_32415.pth /root/ruozhi/huggingface
```
此时，hf 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”。可以简单理解：LoRA 模型文件 = Adapter

## 部署与测试
### 将 HuggingFace adapter 合并到大语言模型：
```shell
# 创建一个存放最终模型的路径
mkdir -p /root/ruozhi/final_model

# 解决一下线程冲突的 Bug 
export MKL_SERVICE_FORCE_INTEL=1

# 进行模型整合
xtuner convert merge /root/ruozhi/model /root/ruozhi/huggingface /root/ruozhi/final_model
```

### 与合并后的模型对话
```shell
# 加载 Adapter 模型对话（Float 16）
xtuner chat /root/ruozhi/final_model --prompt-template internlm2_chat
```
![](13.png)
![](14.png)