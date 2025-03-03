# sd-webui-pe

Tips: English version is at the bottom

For online using, checkout the [HuggingFace Space](https://huggingface.co/spaces/SLAPaper/roborovski-superprompt-v1)

## 2023/11/04 更新

Fooocus V2动态提示功能的Webui移植，安装之后首次运行会自动从 [fooocus_expansion.bin](https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin) 下载GPT-2模型，并保存到插件目录下的 `models/expansion/pytorch_model.bin`

如果自动下载失败，可以自己手动下载。

安装成功只要勾选启动就会生效。
![image](https://github.com/facok/sd-webui-pe/assets/128763816/190e036d-bf40-418b-80eb-14bb1971ca3d)

启用前：
![image](https://github.com/facok/sd-webui-pe/assets/128763816/9f53af4f-2d5c-4490-bcb9-72f43da28416)

启用后：
![image](https://github.com/facok/sd-webui-pe/assets/128763816/39ee44c4-eed3-4e85-b4c8-d3e0692c85f7)

关于Fooocus V2原版功能介绍：
[https://github.com/lllyasviel/Fooocus/discussions/117](https://github.com/lllyasviel/Fooocus/discussions/117#raw)

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

## 2024/03/20 更新

- 修复了 SuperPrompt V1 输出不受种子值影响的问题
- 实现了 DanTagGen-beta 的渐进生成机制，扩展的提示词会更丰富
- 在 `Advanced Option` 中提供了自定义 SuperPrompt V1 使用的提示词的配置

## 2024/05/01 更新

- 将 DanTagGen 支持改为 [HF collections](https://huggingface.co/collections/KBlueLeaf/dantaggen-65f82fa9335881a67573556b) 里的任意版本

## 2025/03/04 更新

新增 TIPO [https://huggingface.co/KBlueLeaf/TIPO-500M-ft](https://huggingface.co/KBlueLeaf/TIPO-500M-ft)

请手动下载 [TIPO-500M-ft-F16.gguf](https://huggingface.co/KBlueLeaf/TIPO-500M-ft/blob/main/TIPO-500M-ft-F16.gguf) 到插件目录下的 `models/TIPO` 下

可以通过启动参数 `--pe-model-path` 来指定模型目录

本模型通用于自然语言和Danbooru 标签的模型

依赖 KGen [https://github.com/KohakuBlueleaf/KGen](https://github.com/KohakuBlueleaf/KGen)

依赖 llama_cpp [https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## Below is English Version

## Update 2023/11/04

WebUI port of Fooocus V2 dynamic prompt expansion. Upon installation and first run, it automatically downloads the GPT-2 model from [fooocus_expansion.bin](https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin) and saves it to `models/expansion/pytorch_model.bin` in the plugin directory.

If the automatic download fails, you can download it manually.

Once installed, simply checking the box to activate it will make it take effect.
![image](https://github.com/facok/sd-webui-pe/assets/128763816/190e036d-bf40-418b-80eb-14bb1971ca3d)

Before enabling:
![image](https://github.com/facok/sd-webui-pe/assets/128763816/9f53af4f-2d5c-4490-bcb9-72f43da28416)

After enabling:
![image](https://github.com/facok/sd-webui-pe/assets/128763816/39ee44c4-eed3-4e85-b4c8-d3e0692c85f7)

For more information on the original Fooocus V2 feature, visit:
[https://github.com/lllyasviel/Fooocus/discussions/117](https://github.com/lllyasviel/Fooocus/discussions/117#raw)

## Update 2024/03/16

Added SuperPrompt V1 [https://huggingface.co/roborovski/superprompt-v1](https://huggingface.co/roborovski/superprompt-v1)

Due to the unstable download function of huggingface_hub/transformers, it needs to be manually cloned to `models/superprompt-v1` in the plugin directory.

The model directory can be specified via the `--pe-model-path` startup parameter.

Differences between the two models:

- Fooocus V2: Only adds some image quality-related prompt words at the end of the original prompt, without modifying the original prompt, thus maintaining the original artistic style.
- SuperPrompt V1: Significantly rewrites the original prompt, potentially resulting in very different visual effects.

## Update 2024/03/19

Added DanTagGen-beta [https://huggingface.co/KBlueLeaf/DanTagGen-beta](https://huggingface.co/KBlueLeaf/DanTagGen-beta)

Due to the unstable download function of huggingface_hub/transformers, it needs to be manually cloned to `models/DanTagGen-beta` in the plugin directory.

The model directory can be specified via the `--pe-model-path` startup parameter.

This model is suitable for models that use danbooru tags, such as Kohaku-XL.

## Update 2024/03/20

- Fixed an issue where SuperPrompt V1 output was not affected by seed values.
- Implemented a progressive generation mechanism for DanTagGen-beta, resulting in richer expanded prompt words.
- Added a custom configuration for prompt words used by SuperPrompt V1 in the `Advanced Option`.

## Update 2024/05/01

- Changed DanTagGen support to any version in [HF collections](https://huggingface.co/collections/KBlueLeaf/dantaggen-65f82fa9335881a67573556b)

## Update 2024/05/27

- Added sampling parameters in Advanced Option, so that you can control the output randomness.

## Update 2025/03/04

Added TIPO [https://huggingface.co/KBlueLeaf/TIPO-500M-ft](https://huggingface.co/KBlueLeaf/TIPO-500M-ft)

Please manually download the file [TIPO-500M-ft-F16.gguf](https://huggingface.co/KBlueLeaf/TIPO-500M-ft/blob/main/TIPO-500M-ft-F16.gguf) and save it to the `models/TIPO` directory under the plugin directory

The model directory can be specified through the startup parameter `--pe-model-path`

This model is universally applicable to both natural language and Danbooru tagging

Depends on KGen [https://github.com/KohakuBlueleaf/KGen](https://github.com/KohakuBlueleaf/KGen)

Dependency on llama_cpp [https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
