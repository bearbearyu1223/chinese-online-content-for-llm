---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [1]"
categories: cs336
author:
- 大模型我都爱
---

<style>
  .xiaohongshu-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #ff2442; /* 小红书主色 */
    text-decoration: none;
    font-weight: bold;
    font-size: 14px;
  }
  .xiaohongshu-link:hover {
    text-decoration: underline;
  }
  .xiaohongshu-logo {
    width: 18px;
    height: 18px;
    border-radius: 4px;
  }
</style>

<div style="padding:12px;border:1px solid #eee;border-radius:8px;display:inline-block;margin-bottom:20px;">
  <strong>大模型我都爱</strong><br>
  <p style="margin:4px 0;">
    小红书号：
    <a class="xiaohongshu-link"
       href="https://www.xiaohongshu.com/user/profile/5b2c5758e8ac2b08bf20e38d"
       target="_blank">
      <img class="xiaohongshu-logo"
           src="https://static.cdnlogo.com/logos/r/77/rednote-xiaohongshu.svg"
           alt="小红书 logo">
      119826921
    </a>
  </p>
  IP属地：美国
</div>

## Introduction

This first post covers **setting up the local development environment**—a small but necessary step to begin engaging with the lecture materials.

这第一篇文章涵盖了**本地开发环境的设置**——这是开始学习课程材料的一个小但必要的步骤。


## Setting Up the Local Dev Environment

## 设置本地开发环境

---

### Prerequisites

Make sure you have Git, Python 3.11, and Node.js installed on your system. You'll also need [`uv`](https://github.com/astral-sh/uv), a fast Python package manager.

### 前提条件

确保你的系统上已安装Git、Python 3.11和Node.js。你还需要[`uv`](https://github.com/astral-sh/uv)，一个快速的Python包管理器。

---

### Step 1: Clone the CS336 Repository

```bash
git clone https://github.com/stanford-cs336/spring2025-lectures
cd spring2025-lectures
```

### 步骤1：Clone the CS336 Repository

```bash
git clone https://github.com/stanford-cs336/spring2025-lectures
cd spring2025-lectures
```

---

### Step 2: Set Up Python Virtual Environment with UV

Install `uv` (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment using Python 3.11:

```bash
uv venv --python=3.11
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate
```

### 步骤2：使用UV设置Python虚拟环境

安装`uv`（如果尚未安装）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

使用Python 3.11创建虚拟环境：

```bash
uv venv --python=3.11
```

激活环境：

```bash
# macOS/Linux
source .venv/bin/activate
```

---

### Step 3: Install Python Dependencies

> 💡 Note: I'm using a MacBook Pro (M4). The `triton` package doesn't support Apple Silicon, so I excluded it during the initial setup.
> More info: [Triton GitHub](https://github.com/triton-lang/triton)

```bash
grep -v "triton" requirements.txt > requirements_no_triton.txt
uv pip install -r requirements_no_triton.txt
```

### 步骤3：安装Python相关库

> 💡 注意：我使用的是MacBook Pro (M4)。`triton`包不支持Apple Silicon，所以我在初始设置时将其排除。
> 更多信息：[Triton GitHub](https://github.com/triton-lang/triton)

```bash
grep -v "triton" requirements.txt > requirements_no_triton.txt
uv pip install -r requirements_no_triton.txt
```

---

### Step 4: Generate Executable Lecture Content

Compile a lecture:

```bash
python execute.py -m lecture_01
```

This will generate a trace file at:

```
var/traces/lecture_01.json
```

### 步骤4：生成可执行的课程内容

编译课程：

```bash
python execute.py -m lecture_01
```

这将在本地以下位置生成一个追踪文件：

```
var/traces/lecture_01.json
```

---

### Step 5: Build the Local React Web App to View Lectures

Install Node.js if needed:

```bash
brew install node
```

Then build and serve the trace viewer:

```bash
cd trace-viewer
npm install
npm run dev
```

Open the viewer in your browser:

```
http://localhost:<PORT>?trace=var/traces/lecture_01.json
```

It should be something like ![this]({{ site.baseurl }}/assets/picture/2025_07_20_cs336_note_get_started/cs336_lecture_view.png).

### 步骤5：构建本地React Web应用以查看课程

如果需要，安装Node.js：

```bash
brew install node
```

然后构建并运行追踪查看器：

```bash
cd trace-viewer
npm install
npm run dev
```

在浏览器中打开查看器：

```
http://localhost:<PORT>?trace=var/traces/lecture_01.json
```

本地服务端应该看起来像![这样]({{ site.baseurl }}/assets/picture/2025_07_20_cs336_note_get_started/cs336_lecture_view.png)。

---

Then enjoy going through the lecture notes!

然后享受学习课程笔记的过程吧！
