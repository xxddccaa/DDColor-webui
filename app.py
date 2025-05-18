import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import argparse
import requests
import io

class DDColorClient:
    def __init__(self, api_url="http://localhost:9701/colorize"):
        """Initialize client for the DDColor API"""
        self.api_url = api_url
        print(f"API 连接地址: {api_url}")
    
    def colorize(self, input_image):
        """Send image to API for colorization and return results"""
        if input_image is None:
            return None, None, None
        
        # 如果输入是文件路径，则打开图像
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        try:
            # 准备图像用于API请求
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # 发送API请求
            files = {'file': ('image.png', img_byte_arr, 'image/png')}
            response = requests.post(self.api_url, files=files, timeout=60)
            
            # 检查响应状态
            if response.status_code != 200:
                error_message = f"API请求失败: {response.status_code}"
                try:
                    error_detail = response.json().get('detail', '')
                    if error_detail:
                        error_message += f": {error_detail}"
                except:
                    pass
                raise Exception(error_message)
            
            # 获取RGB格式的原始图像
            rgb_img = np.array(input_image)
            
            # 创建LAB色彩空间的L通道灰度图
            lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            l_channel = lab_img[:, :, 0]
            
            # 创建灰度图像用于显示（3通道）
            grayscale_img = np.repeat(l_channel[:, :, np.newaxis], 3, axis=2)
            grayscale_img = ((grayscale_img / 100) * 255).astype(np.uint8)
            
            # 从API响应中获取上色后的图像
            colorized_img = Image.open(io.BytesIO(response.content))
            colorized_rgb = np.array(colorized_img)
            
            return rgb_img, grayscale_img, colorized_rgb
            
        except Exception as e:
            print(f"上色过程中出错: {str(e)}")
            return None, None, None


def create_gradio_interface(client):
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
            fn=client.colorize,
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
    parser = argparse.ArgumentParser(description='黑白图上色应用 (LAB-L通道)')
    parser.add_argument('--api_url', type=str, default='http://localhost:9701/colorize',
                      help='DDColor API服务地址')
    parser.add_argument('--port', type=int, default=7862,
                      help='Gradio应用的端口号')
    parser.add_argument('--share', action='store_true',
                      help='生成可分享的链接')
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建API客户端
    client = DDColorClient(api_url=args.api_url)
    
    print(f"正在连接黑白图上色API服务: {args.api_url}")
    print(f"注意：本应用使用LAB色彩空间的L通道进行灰度转换")
    
    # 创建Gradio界面
    app = create_gradio_interface(client)
    
    # 启动应用
    app.launch(server_port=args.port, server_name="0.0.0.0", share=args.share)
    

if __name__ == "__main__":
    main()