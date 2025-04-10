import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import argparse
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from skimage import color

class DDColorModelLab:
    def __init__(self, model_path, cuda=True):
        """初始化DDColor上色模型，使用LAB色彩空间的L通道"""
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 初始化modelscope管道
        self.model = pipeline(
            Tasks.image_colorization,
            model=model_path,
            device=self.device
        )
        
        print(f"黑白图上色模型加载成功，路径: {model_path}!")
    
    def process_image(self, image):
        """处理上传的图像：使用L通道转为灰度图并进行着色"""
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取RGB格式的numpy数组
        rgb_img = np.array(image)
        
        # 使用LAB色彩空间的L通道创建灰度版本
        lab_img = color.rgb2lab(rgb_img)
        l_channel = lab_img[:, :, 0]
        
        # 创建灰度图像用于显示（3通道）
        grayscale_img = np.repeat(l_channel[:, :, np.newaxis], 3, axis=2)
        grayscale_img = ((grayscale_img + 100) / 200 * 255).astype(np.uint8)
        
        # 保存原始图像到临时文件（modelscope管道需要文件路径）
        temp_path = 'temp_input.jpg'
        Image.fromarray(rgb_img).save(temp_path)
        
        # 运行modelscope管道进行着色
        result = self.model(temp_path)
        
        # 获取输出的着色图像（ModelScope已经输出BGR格式）
        colorized_bgr = result[OutputKeys.OUTPUT_IMG]
        
        # 将BGR转换为RGB用于Gradio显示
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return rgb_img, grayscale_img, colorized_rgb
    
    def colorize(self, input_image):
        """Gradio界面的着色函数"""
        if input_image is None:
            return None, None, None
        
        # 如果输入是文件路径，则打开图像
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # 处理图像
        original, grayscale, colorized = self.process_image(input_image)
        
        return original, grayscale, colorized


def create_gradio_interface(model):
    """创建Gradio界面"""
    with gr.Blocks(title="黑白图上色模型 (LAB-L通道)") as app:
        gr.Markdown("# 黑白图上色模型 (使用LAB色彩空间的L通道)")
        gr.Markdown("上传一张图片，系统将使用L通道灰度化并自动为其上色")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传图片", type="pil")
                colorize_btn = gr.Button("开始上色", variant="primary")
            
        with gr.Row():
            with gr.Column(scale=1):
                original_image = gr.Image(label="原始图片")
            with gr.Column(scale=1):
                grayscale_image = gr.Image(label="灰度图片 (L通道)")
            with gr.Column(scale=1):
                colorized_image = gr.Image(label="上色结果")
        
        colorize_btn.click(
            fn=model.colorize,
            inputs=[input_image],
            outputs=[original_image, grayscale_image, colorized_image]
        )
        
        gr.Markdown("## 使用说明")
        gr.Markdown("1. 点击上方图片区域上传图片或者拖放图片")
        gr.Markdown("2. 点击开始上色按钮")
        gr.Markdown("3. 等待几秒钟，系统将显示原始图片、L通道灰度版本和上色结果")
        gr.Markdown("4. 如果您对结果不满意，可以上传新图片再次尝试")
        gr.Markdown("5. 本版本使用LAB色彩空间的L通道进行灰度转换，而不是标准的RGB转灰度方法")
        
    return app


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='黑白图上色模型 (使用LAB-L通道)')
    parser.add_argument('--model_path', type=str, default='./DDColormodel',
                      help='DDColor模型目录的路径')
    parser.add_argument('--cpu', action='store_true',
                      help='强制使用CPU而不是GPU')
    parser.add_argument('--port', type=int, default=7861,
                      help='Gradio应用的端口号')
    parser.add_argument('--share', action='store_true',
                      help='生成可分享的链接')
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置模型路径
    model_path = args.model_path
    
    # 检测CUDA是否可用
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    print(f"正在加载黑白图上色模型，路径: {model_path}")
    print(f"CUDA可用: {use_cuda}")
    print(f"注意：本版本使用LAB色彩空间的L通道进行灰度转换")
    
    # 加载模型
    colorize_model = DDColorModelLab(model_path, use_cuda)
    
    # 创建Gradio界面
    app = create_gradio_interface(colorize_model)
    
    # 启动应用
    app.launch(server_port=args.port, server_name="0.0.0.0", share=args.share)
    

if __name__ == "__main__":
    main() 