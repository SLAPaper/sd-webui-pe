# sd-webui-pe

## 2023/11/04 更新

Fooocus V2动态提示功能的Webui移植，安装之后首次运行会自动从 https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin 下载GPT-2模型，并保存到插件目录下的 `models/expansion/pytorch_model.bin`

如果自动下载失败，可以自己手动下载。

安装成功只要勾选启动就会生效。
![image](https://github.com/facok/sd-webui-pe/assets/128763816/190e036d-bf40-418b-80eb-14bb1971ca3d)

启用前：
![image](https://github.com/facok/sd-webui-pe/assets/128763816/9f53af4f-2d5c-4490-bcb9-72f43da28416)

启用后：
![image](https://github.com/facok/sd-webui-pe/assets/128763816/39ee44c4-eed3-4e85-b4c8-d3e0692c85f7)

关于Fooocus V2原版功能介绍：
https://github.com/lllyasviel/Fooocus/discussions/117#raw


## 2024/03/16 更新

新增 SuperPrompt V1 [https://huggingface.co/roborovski/superprompt-v1](https://huggingface.co/roborovski/superprompt-v1)

由于huggingface_hub/transformers的下载功能不稳定，需要手动clone到插件目录下的 `models/superprompt-v1`

可以通过启动参数 `--pe-model-path` 来指定模型目录

两种模型的区别：

- Fooocus V2: 只会在原提示词的结尾添加一些画质相关的提示词，不会对原提示词进行修改，因此会忠于原提示词的画面风格
- SuperPrompt V1: 会对原提示词进行大幅重写，可能会得到非常不同的画面效果


## 2024/03/19 更新

新增 DanTagGen-beta [https://huggingface.co/KBlueLeaf/DanTagGen-beta](https://huggingface.co/KBlueLeaf/DanTagGen-beta)

由于huggingface_hub/transformers的下载功能不稳定，需要手动clone到插件目录下的 `models/DanTagGen-beta`

可以通过启动参数 `--pe-model-path` 来指定模型目录

本模型适用于 Kohaku-XL 等使用 danbooru 标签的模型
