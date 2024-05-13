# 设计思路

这是一个简单的对facefusion的简化工作。主要功能就是实现facefusion中的
```
face_debugger
face_swapper
face_enhander
face_colorizer
frame_enhander
```
这几个功能。但会简化很多

## face_debugger
在facefusion中，face_debugger其实在导入图片或者视频的时候就已经实现了，是通过face_analyser,face_helper,face_masker,face_store等实现的，这也是其他所有功能的基础，而face_debugger其实只是画了个图而已。
## face_swapper
face_swapper默认模型是inswapper_128_fp16.onnx，所以我也打算用这个模型。模型的用法很简单，输入source，一般是原图片的embedding数据，这个是通过另一个模型得到的。而target就是要换的脸。然后执行run就可以了。
## face_colorizer
face_colorizer默认模型是ddcolor，功能就是重建色彩，输入一张黑白照可以重建照片色彩。
## face_enhander
face_enhander默认模型是gfpgan_1.4，功能是脸部增强。但这个模型除了快效果好像没有codeformer好，所以我打算使用codeformer
## frame_enhander
frame_enhander默认模型是span_kendata_x4，功能是高清放大，它是将模型分为一个个256或者512的小块，对每个小块进行方法，然后将小块拼接一起。因为拼接，所以在拼接处是有明显分界的，所以还要混合一下。只需要将原始图片放大，然后两个加权平均一下就可以了。

## core
这里的core文件是指facefusion/core.py文件,不是其他的core文件。和名字一样，core文件是facefusion的核心，它是统一前后端的文件。通过conditional_process()函数，可以让前端gradio来执行执行图像或视频处理命令。而图像处理是通过process_image(start_time)函数，视频处理是通过process_video(start_time)函数。但这两个函数也不是直接处理图像或者视频的，他们也是通过调用模块中的process_image或process_video来处理的。

所以core就是中间核心，起到连接前后端的作用。

## process_manager
这个文件规定了执行的状态。整个程序有四种状态：`pending`,"checking","processing","stopping"。不同状态下执行不同的任务，明确了当前任务情况。
另外，通过manage函数，它可以中断当前的任务。

## core
这里的core文件是指frame下的core.py文件。它规定了所有模块的执行方式和必要函数。
所有的模块中视频都是通过multi_process_frames函数执行的，并非直接执行。
get_frame_processors_modules则可以导入指定的模块，在facefusion.core中就通过这个函数来指定都执行哪些模块。
core规定每个模块都必须有以下方法
```
'get_frame_processor', 载入onnx模型
'clear_frame_processor', 清理模型
'get_options', 得到模型，包括type，url，path，size等信息
'set_options', 设置模型
'register_args',注册参数，在facefusion.core中调用
'apply_args', 应用参数，在facefusion.core中调用
'pre_check', 预检查，在facefusion.core中调用
'post_check', 后检查，在facefusion.core中调用
'pre_process', 预处理，在facefusion.core中调用
'post_process', 后处理，在facefusion.core中调用
'get_reference_frame', 得到参考帧,大多情况下没用
'process_frame', 主要的处理函数，模块的具体执行逻辑就在这里
'process_frames', 多帧处理函数，
'process_image', 处理图像，在facefusion.core中调用
'process_video', 处理视频，在facefusion.core中调用
```
因为我只想做一个简化的程序，所以这些函数我就只保留其中部分。

# 这个程序的设计

这个程序设计简单一些，分为前中后三端。
前端:fornt_end 就是通过gradio来展示界面.
后端:back_end 就是具体的模型实现代码.
中端:mid_end 就是连接前后端.

# 后端
后端先设计face_analyser,这是换脸模块实现的前提。这个功能通过四个模型实现
```
"face_detectors": yoloface_8n.onnx, 用于探测脸部
"face_recognizer": arcface_w600k_r50.onnx，用于识别脸部
"face_landmarkers": 2dfan4.onnx, 用于获取人脸特征点
"gender_age": gender_age.onnx, 用户识别人物年龄性别
```
face_analyser的目的是得到图像中的每个脸，包括脸的大小，位置，五官，遮罩，编码等等数据。

数据设计：所有的数据不是图像就是视频，那么本质上就可以看做图像。facefusion中将所有数据看做帧frame，我也用同样的设计吧。




