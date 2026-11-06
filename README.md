深度学习目标检测中利用yolov5训练垃圾检测数据集 

基于Yolov5的智能垃圾识别系统
1.描述：利用pytorch模型，采用Yolov5技术，致力于实现对垃圾的智能识别与分类系统
2.算法:Yolov5+Flask+vue
@[toc]
以下文字及代码仅供参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5f7e526e7d44760a5fb533353a3510a.png)


1.项目描述：利用pytorch模型，采用Yolov5技术，致力于实现对垃圾的智能识别与分类系统
2.采用技术:Yolov5+Flask+vue
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e703be6c082342b4bfdbc612ebf85dfb.png)
1
以下是一个基于 **YOLOv5** 的智能垃圾识别系统的实现，结合 **Flask** 和 **Vue.js** 前后端分离架构的详细代码和步骤。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54d050c7fbe3437f915f7d086601eeff.png)

## **项目结构**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/462b14f0f0d24ca4a2a62214f8b1f252.png)

```
smart-trash-recognition/
├── backend/               # Flask 后端服务
│   ├── app.py             # Flask 主程序
│   ├── requirements.txt   # 依赖库
│   └── weights/           # YOLOv5 模型权重文件
├── frontend/              # Vue.js 前端界面
│   ├── src/               # Vue.js 源码
│   │   ├── App.vue        # 主组件
│   │   ├── main.js        # 入口文件
│   │   └── api.js         # API 请求封装
│   └── package.json       # Vue.js 依赖配置
└── dataset/               # 数据集（YOLO 格式）
    ├── images/            # 图片数据
    └── labels/            # 标注文件
```

---

## **一、安装依赖**

### **1. 安装 Python 环境**
确保已安装 Python（推荐版本 3.8+），并安装以下依赖：

```bash
pip install torch torchvision flask opencv-python pillow
```

- `torch` 和 `torchvision`：深度学习框架。
- `flask`：用于构建后端 API。
- `opencv-python`：图像处理。
- `pillow`：图片加载与处理。

### **2. 安装 YOLOv5**
克隆 YOLOv5 官方仓库并安装依赖：

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

将训练好的模型权重文件（如 `best.pt`）放置在 `backend/weights/` 目录中。

---

## **二、准备数据集**
同学呀：
假设你的数据集已经按照 YOLO 格式组织好：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

创建 `data.yaml` 文件，内容如下：

```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 6                   # 类别数量（示例：6类垃圾）
names: ['plastic', 'metal', 'paper', 'glass', 'organic', 'other']  # 类别名称
```

训练模型：

```python
from yolov5 import train

# 训练模型
train.run(data='path/to/dataset/data.yaml', epochs=50, imgsz=640, batch=16)
```

训练完成后，生成的模型权重文件（如 `best.pt`）将保存在 `runs/train/` 目录下。

---

## **三、Flask 后端服务**

### **1. 创建 Flask 应用**

在 `backend/app.py` 中编写以下代码：

```python
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 读取图片
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 进行目标检测
    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient='records')

    # 返回检测结果
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### **2. 启动 Flask 服务**

在终端中运行以下命令启动 Flask 服务：

```bash
cd backend
python app.py
```

---

## **四、Vue.js 前端界面**

### **1. 初始化 Vue.js 项目**

使用 Vue CLI 初始化前端项目：

```bash
npm install -g @vue/cli
vue create frontend
cd frontend
npm install axios
```

### **2. 编写前端代码**

#### **2.1 `src/main.js`**

```javascript
import { createApp } from 'vue'
import App from './App.vue'

createApp(App).mount('#app')
```

#### **2.2 `src/api.js`**

封装 API 请求：

```javascript
import axios from 'axios'

const BASE_URL = 'http://localhost:5000'

export const uploadImage = (file) => {
  const formData = new FormData()
  formData.append('file', file)

  return axios.post(`${BASE_URL}/detect`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
```

#### **2.3 `src/App.vue`**

```vue
<template>
  <div id="app">
    <h1>智能垃圾识别系统</h1>
    <input type="file" @change="handleFileUpload" accept="image/*" />
    <img v-if="imageUrl" :src="imageUrl" alt="Uploaded Image" width="400" />
    <div v-if="results.length">
      <h2>检测结果：</h2>
      <ul>
        <li v-for="(item, index) in results" :key="index">
          {{ item.name }} (置信度: {{ (item.confidence * 100).toFixed(2) }}%)
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import { uploadImage } from './api'

export default {
  data() {
    return {
      imageUrl: null,
      results: [],
    }
  },
  methods: {
    async handleFileUpload(event) {
      const file = event.target.files[0]
      if (!file) return

      this.imageUrl = URL.createObjectURL(file)

      try {
        const response = await uploadImage(file)
        this.results = response.data.detections.map((item) => ({
          name: item.name,
          confidence: item.confidence,
        }))
      } catch (error) {
        console.error('Error uploading image:', error)
      }
    },
  },
}
</script>

<style>
#app {
  font-family: Arial, sans-serif;
  text-align: center;
  margin-top: 50px;
}
</style>
```

### **3. 启动 Vue.js 前端**

在终端中运行以下命令启动前端服务：

```bash
cd frontend
npm run serve
```

访问 `http://localhost:8080` 即可使用前端界面上传图片并查看检测结果。

---

## **五、总结**

同学呀，完成一个基于 YOLOv5 的智能垃圾识别系统，包括：

1. **后端服务**：使用 Flask 提供 RESTful API 接口，接收图片并返回检测结果。
2. **前端界面**：使用 Vue.js 实现图片上传和结果展示。

