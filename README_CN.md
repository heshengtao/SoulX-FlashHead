<div align="center">


<h1>SoulX-FlashHead: Oracle-guided Generation of Infinite Real-time Streaming Talking Heads</h1>

[Tan Yu*](https://jiayoujiayoujiayoua.github.io/), [Qian Qiao*](https://qianqiaoai.github.io/)<sup>&#9993;</sup>, [Le Shen*](https://openreview.net/profile?id=%7ELe_Shen3), [Ke Zhou](https://github.com/jokerz0624), [Jincheng Hu](#), [Dian Sheng](#), [Bo Hu](#), [Haoming Qin](#), [Jun Gao](#), [Changhai Zhou](#), [Shunshun Yin](#), [Siyuan Liu](#) <sup>&#9993;</sup>


<sup>*</sup>共同一作
<sup>&#9993;</sup>通讯作者


<a href='https://soul-ailab.github.io/soulx-flashhead/' target="_blank"><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://arxiv.org/pdf/2602.07449' target="_blank"><img src='https://img.shields.io/badge/Technical-Report-red'></a>
<a href='https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B' target="_blank"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href="https://huggingface.co/datasets/Soul-AILab/VividHead" target="_blank"><img src="https://img.shields.io/badge/🤗 Hugging Face-Dataset-blue" alt="Dataset"></a>&nbsp;
</div>

<p align="right"><a href="./README.md">English</a> | <b>中文</b></p>

> **🤖 给用户和 AI Agent 的重要提示**
>
> 本项目包含一份专为 AI Agent 编写的自动化部署文档：**[DEPLOYMENT_FOR_AGENTS.md](./DEPLOYMENT_FOR_AGENTS.md)**。
> 如果你希望让 AI（如 OpenCode / Claude / Copilot 等）自动完成本项目的部署，请把该文档交给 AI 阅读并让它严格按步骤执行；人工部署也请优先参考该文档中的验证清单与故障排查表。

## 🧩 本仓库的集成功能

> 以下为本仓库在官方版本基础上增加的集成功能，主要用于配合 [super-agent-party](https://github.com/heshengtao/super-agent-party) 桌宠。

- **多参考图目录模式（Directory Mode）**：`init` 消息的 `cond_image` 支持传入**目录路径**（`cond_is_path: true`），服务端会加载目录下全部 `*.png` 作为多个人物形象；之后发送 `{"type":"reset","person_name":"<文件名去扩展名>"}` 即可**热切换人物**，无需重新初始化。
- **透明背景 / 服务端抠图（RVM Matting）**：`init` 消息新增 `transparent_bg: bool`（或设置环境变量 `FLASHHEAD_MATTING=1`）。启用后服务端使用 RVM (Robust Video Matting, MobileNetV3) 对每帧实时抠图，帧编码从 JPEG 切换为 **WebP (RGBA 带 alpha)**；`frames_meta` 消息新增 `fmt` 字段（`jpeg` / `webp`）告知客户端当前编码格式。抠图模型在首次启用时通过 torch.hub 自动下载（约 14.5MB），失败自动回退 JPEG。
- **Windows 路径修复**：目录模式下 person_name 解析在 Windows 上原本会得到完整路径，已修复为 `os.path.basename`。

协议细节见 [API.md](./API.md) 与 [DEPLOYMENT_FOR_AGENTS.md](./DEPLOYMENT_FOR_AGENTS.md)。

## ⚡ 亮点
- **Model_Lite** [已发布](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B/tree/main/Model_Lite)：单张 RTX4090 可达 96 FPS，或 3 路并发实时（25+ FPS）流式生成。
- **Model_Pro** [已发布](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B/tree/main/Model_Pro)：单张 RTX4090 可生成 10.8 FPS 高质量视频，双 RTX5090 可实时（25+ FPS）。
- **Model_Pretrained** 即将发布，为社区研究提供高性能权重与实验基础。

## 🔥 更新动态
- **2026.03.09** - HuggingFace 在线 Demo 已上线，可直接[体验](https://huggingface.co/spaces/Soul-AILab/SoulX-FlashHead)。
- **2026.03.04** - Gradio 应用已发布，普通模式与流式模式均支持。
- **2026.03.02** - [ComfyUI 节点](https://github.com/HM-RunningHub/ComfyUI_RH_FlashHead)已可用，感谢 [HM-RunningHub](https://github.com/HM-RunningHub) 的 ComfyUI 支持。
- **2026.02.12** - [在线体验](#在线体验二维码) 已通过 Soul App 上线，立即下载体验。
- **2026.02.12** - 我们发布了[推理代码](https://github.com/Soul-AILab/SoulX-FlashHead)与[模型权重](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)。
- **2026.02.12** - 我们在 [SoulX-FlashHead](https://soul-ailab.github.io/soulx-flashhead/) 发布了**项目主页**。
- **2026.02.07** - 我们发布了[数据集](https://huggingface.co/datasets/Soul-AILab/VividHead)。
- **2026.02.07** - 我们在 [Arxiv](https://arxiv.org/pdf/2602.07449) 与 [GitHub 仓库](./assets/SoulX_FlashHead.pdf) 发布了 **SoulX-FlashHead 技术报告**。


## 📑 任务清单
- [x] 技术报告
- [x] 项目主页
- [x] 推理代码
- [x] HuggingFace 流式在线 Demo
- [x] Pro 模型与 Lite 模型蒸馏权重发布
- [ ] 预训练权重发布

## 🌰 示例
更多示例见项目仓库。

<table>
  <tbody>
    <!-- Row 1: Videos 1-5 -->
    <tr>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/548713532-5d4800cf-d0dd-4aaf-a887-d9f202d3b4b6.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA4ODYxMTcsIm5iZiI6MTc3MDg4NTgxNywicGF0aCI6Ii8xNzYzOTE0MjQvNTQ4NzEzNTMyLTVkNDgwMGNmLWQwZGQtNGFhZi1hODg3LWQ5ZjIwMmQzYjRiNi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIxMlQwODQzMzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xZWQzODMzYjYzZmE1ODc2ZjA0NDhkYzcyZGIxZDRiYzlmNTU0M2Y1ZGUxNjlmYzgzMjNhMTM1MTQ2MGNmMzM1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.e7zRf7beypjK6JnJpUnivJv-1_s937aK89TxTa2m_Sc" style="width:100%; aspect-ratio:512/512; object-fit:cover;" controls loop></video></td>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/548713758-051f5779-cd5d-4336-9326-3b2e55ccc77d.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA4ODYxNDAsIm5iZiI6MTc3MDg4NTg0MCwicGF0aCI6Ii8xNzYzOTE0MjQvNTQ4NzEzNzU4LTA1MWY1Nzc5LWNkNWQtNDMzNi05MzI2LTNiMmU1NWNjYzc3ZC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIxMlQwODQ0MDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mZDdlNTA4YjJjNmQzZGZlNWQwYjc0MmZkOTcyYWFhZjY2NzRjMjQyNjI3YWMyZDA5ZmI0ZGNiZDc0ODBhYTg2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.S5cJs9MLQRXKUQsC2lCZl74QQKI2orWxT-NcF4qHpr0" style="width:100%; aspect-ratio:512/512; object-fit:cover;" controls loop></video></td>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/548713661-8cdfd881-7782-403c-9dc0-e93930750dfe.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA4ODYxMzEsIm5iZiI6MTc3MDg4NTgzMSwicGF0aCI6Ii8xNzYzOTE0MjQvNTQ4NzEzNjYxLThjZGZkODgxLTc3ODItNDAzYy05ZGMwLWU5MzkzMDc1MGRmZS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIxMlQwODQzNTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xZWQzODMzYjYzZmE1ODc2ZjA0NDhkYzcyZGIxZDRiYzlmNTU0M2Y1ZGUxNjlmYzgzMjNhMTM1MTQ2MGNmMzM1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.blrq-obdV5NBPojgWEdOaXCVriCAIZjsKQ_x_DDPQ5k" style="width:100%; aspect-ratio:512/512; object-fit:cover;" controls loop></video></td>
    </tr>

  </tbody>
</table>



## 📖 快速开始
###  🔧 安装
#### 1. 创建 Conda 环境
```bash
conda create -n flashhead python=3.10
conda activate flashhead
```
#### 2. 安装 CUDA 版 PyTorch
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```
#### 3. 安装其他依赖
```bash
pip install -r requirements.txt
```
#### 4. 安装 FlashAttention：
```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

-- 如果编译耗时过长，推荐以下方式：
1. 从[此处](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)下载 wheel 文件
2. pip install xxx.whl

#### 5. 安装 SageAttention（可选）
```bash
pip install sageattention==2.2.0 --no-build-isolation
```

#### 6. 安装 FFmpeg
```bash
# Ubuntu / Debian
apt-get install ffmpeg
# CentOS / RHEL
yum install ffmpeg ffmpeg-devel
```
或
```bash
# Conda（无需 root）
conda install -c conda-forge ffmpeg==7
```
### 🤗 模型下载
| 模型组件 | 说明 | 链接 |
| :--- | :--- | :---: |
| `SoulX-FlashHead-1_3B` | 我们的 1.3B 模型 | 🤗 [Huggingface](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | wav2vec2-base-960h | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h) |

```bash
# 如果你在中国大陆，先执行: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```
### 🚀 推理
```bash
# 单卡 [Pro 模型] 推理
bash inference_script_single_gpu_pro.sh


# 多卡 [Pro 模型] 推理
bash inference_script_multi_gpu_pro.sh
# Pro 模型的实时推理速度需要两张 RTX-5090 并配合 SageAttention。


# 单卡 [Lite 模型] 推理
bash inference_script_single_gpu_lite.sh
# 单张 RTX-4090 即可支持实时推理（最高 3 路并发）。
```

### ⚡️ Gradio 演示
```bash
# Gradio 支持需要 gradio==5.50.0，推荐使用 Chrome。

# 普通 gradio demo
python gradio_app.py

# 流式 gradio demo（仅支持单卡）
python gradio_app_streaming.py
```

### 🤗 流式在线 Demo
点击[这里](https://huggingface.co/spaces/Soul-AILab/SoulX-FlashHead)在 HuggingFace Spaces 上体验实时流式 Demo。


### 👋 在线体验
扫码进入活动链接，体验实时互动。[2026.2.12~2026.3.11]
<a id="在线体验二维码"></a>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/soul_event_link.png" width="200" alt="SoulApp event QR Code"/>
        <br />
        <strong>Real-time Online Experience<br>(SoulApp 实时在线体验)</strong>
      </td>
    </tr>
  </table>
</div>

## 📧 联系我们
如果你对我们的工作有任何想法或建议，欢迎邮件联系 yutan@soulapp.cn 或 qiaoqian@soulapp.cn 或 le.shen@mail.dhu.edu.cn 或 zhouke@soulapp.cn 或 liusiyuan@soulapp.cn

我们开通了微信群。同时，我们代表 **SoulApp** 诚挚欢迎大家下载 App 并加入 Soul 群组，参与更深入的技术讨论并获取最新动态！

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/wechat_group.png" width="300" alt="WeChat Group QR Code"/>
        <br />
        <strong>Join WeChat Group<br>(加入微信技术群)</strong>
      </td>
      <td width="100"></td>
      <td align="center">
        <img src="assets/soul_group.png" width="300" alt="Soul App Group QR Code"/>
        <br />
        <strong>Download SoulApp & Join Group<br>(下载SoulApp加入群组)</strong>
      </td>
    </tr>
  </table>
</div>

 
## 📚 引用

如果我们的工作对你的研究有帮助，请考虑引用：

```
@article{yu2026soulx,
  title={SoulX-FlashHead: Oracle-guided Generation of Infinite Real-time Streaming Talking Heads},
  author={Yu, Tan and Qiao, Qian and Shen, Le and Zhou, Ke and Hu, Jincheng and Sheng, Dian and Hu, Bo and Qin, Haoming and Gao, Jun and Zhou, Changhai and others},
  journal={arXiv preprint arXiv:2602.07449},
  year={2026}
}
```

## 🙇 致谢
- [Wan](https://github.com/Wan-Video/Wan2.1)：我们基于的基础模型。
- [LTX-Video](https://github.com/Lightricks/LTX-Video)：Lite 模型所使用的 VAE。
- [Self forcing](https://github.com/guandeh17/Self-Forcing)：我们基于的代码库。
- [DMD](https://github.com/tianweiy/DMD2) 与 [Self forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus)：我们方法所使用的关键蒸馏技术。
- [SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/)：我们团队的另一款模型，拥有 14B 参数并具备实时能力。
> [!TIP]
> 如果我们的工作对你有帮助，也请考虑为这些基础方法的原始仓库点个 Star。

## 💡 Star History
<p align="center">
  <a href="https://star-history.com/#Soul-AILab/SoulX-FlashHead&Date">
    <img src="https://api.star-history.com/svg?repos=Soul-AILab/SoulX-FlashHead&type=Date" alt="Star History Chart" width="100%">
  </a>
</p>
