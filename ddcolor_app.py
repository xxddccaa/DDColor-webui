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

class DDColorModel:
    def __init__(self, model_path, cuda=True):
        """Initialize the DDColor colorization model"""
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # Initialize the modelscope pipeline
        self.model = pipeline(
            Tasks.image_colorization,
            model=model_path,
            device=self.device
        )
        
        print(f"黑白图上色模型加载成功，路径: {model_path}!")
    
    def process_image(self, image):
        """Process the uploaded image: convert to grayscale and then colorize"""
        # Ensure image is RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get RGB numpy array from PIL image
        rgb_img = np.array(image)
        
        # Create grayscale version for display
        grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
        
        # Save the image to a temporary file (modelscope pipeline expects a path)
        temp_path = 'temp_input.jpg'
        Image.fromarray(rgb_img).save(temp_path)
        
        # Run colorization with modelscope pipeline
        result = self.model(temp_path)
        
        # Get the output colorized image (already in BGR format from ModelScope)
        colorized_bgr = result[OutputKeys.OUTPUT_IMG]
        
        # Convert BGR to RGB for display in Gradio
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return rgb_img, grayscale_img, colorized_rgb
    
    def colorize(self, input_image):
        """Function for Gradio interface"""
        if input_image is None:
            return None, None, None
        
        # If input is a file path, open the image
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Process the image
        original, grayscale, colorized = self.process_image(input_image)
        
        return original, grayscale, colorized


def create_gradio_interface(model):
    """Create Gradio interface"""
    with gr.Blocks(title="黑白图上色模型") as app:
        gr.Markdown("# 黑白图上色模型")
        gr.Markdown("上传一张图片，系统将自动为其上色")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传图片", type="pil")
                colorize_btn = gr.Button("开始上色", variant="primary")
            
        with gr.Row():
            with gr.Column(scale=1):
                original_image = gr.Image(label="原始图片")
            with gr.Column(scale=1):
                grayscale_image = gr.Image(label="灰度图片")
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
        gr.Markdown("3. 等待几秒钟，系统将显示原始图片、灰度版本和上色结果")
        gr.Markdown("4. 如果您对结果不满意，可以上传新图片再次尝试")
        
    return app


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='黑白图上色模型')
    parser.add_argument('--model_path', type=str, default='./DDColormodel',
                      help='Path to the DDColor model directory')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU instead of GPU')
    parser.add_argument('--port', type=int, default=7860,
                      help='Port number for the Gradio app')
    parser.add_argument('--share', action='store_true',
                      help='Generate a shareable link')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set model path
    model_path = args.model_path
    
    # Detect if CUDA is available
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    print(f"正在加载黑白图上色模型，路径: {model_path}")
    print(f"CUDA可用: {use_cuda}")
    
    # Load model
    colorize_model = DDColorModel(model_path, use_cuda)
    
    # Create Gradio interface
    app = create_gradio_interface(colorize_model)
    
    # Launch the app
    app.launch(server_port=args.port, server_name="0.0.0.0", share=args.share)
    

if __name__ == "__main__":
    main()