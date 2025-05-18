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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uvicorn
import io

# Create FastAPI app
app = FastAPI(title="DDColor API", description="API for black and white image colorization using DDColor model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # 将BGR转换为RGB用于显示
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return rgb_img, grayscale_img, colorized_rgb

# Global model instance
model = None

def load_model(model_path, use_cuda=True):
    """Load the DDColor model"""
    global model
    model = DDColorModelLab(model_path, use_cuda)
    return model

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = os.environ.get("MODEL_PATH", "./DDColormodel")
    use_cuda = os.environ.get("USE_CUDA", "1") == "1"
    load_model(model_path, use_cuda)

@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...)):
    """API endpoint to colorize an uploaded image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image
        _, _, colorized = model.process_image(image)
        
        # Convert to bytes for response
        img_byte_arr = io.BytesIO()
        Image.fromarray(colorized).save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return image as response
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DDColor API Server')
    parser.add_argument('--model_path', type=str, default='/DDColormodel',
                      help='Path to DDColor model directory')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage instead of GPU')
    parser.add_argument('--port', type=int, default=9701,
                      help='API server port')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables for model loading
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["USE_CUDA"] = "0" if args.cpu else "1"
    
    print(f"Starting API server on port {args.port}")
    print(f"Model path: {args.model_path}")
    print(f"Using CUDA: {not args.cpu and torch.cuda.is_available()}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)