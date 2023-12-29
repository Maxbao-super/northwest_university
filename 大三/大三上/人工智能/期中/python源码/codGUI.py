import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from AnimalIdentify import identifyImage

animal = {0: "熊", 1: "猫", 2: "狗", 3: "鸭子", 4: "鹅", 5: "熊猫", 6: "猪", 7: "羊", 8: "虎", 9: "人"}


def upload_image():
    file_path = filedialog.askopenfilename()
    global preImage
    if file_path:
        # 打开图像文件
        image = Image.open(file_path)
        preImage = image
        # 调整图像大小以适应预览区域
        image = image.resize((512, 512))
        # 将图像转换为Tkinter可用的格式
        photo = ImageTk.PhotoImage(image)
        # 在标签上显示图像
        image_label.configure(image=photo)
        image_label.image = photo
        # 清除预测内容
        updateText("")


def updateText(text):
    # 在文本区域显示图像名称
    name_text.configure(state="normal")
    name_text.delete('1.0', tk.END)
    name_text.insert(tk.END, text)
    name_text.configure(state="disabled")


def recognize_image():
    # 在这里编写识别图像的代码
    recognize = identifyImage.preIdentify(preImage)
    updateText(animal[recognize])
    pass


def exit_app():
    root.destroy()


# 创建主窗口
root = tk.Tk()
root.title("Animals")
root.geometry('1280x720')
preImage = ""

# 创建上传按钮
upload_button = tk.Button(root, text='上传图片', command=upload_image, width=15, height=3)

upload_button.place(x=300, y=600)

# 创建识别按钮
recognize_button = tk.Button(root, text='识别', command=recognize_image, width=15, height=3)
recognize_button.place(x=950, y=400)

# 创建退出按钮
exit_button = tk.Button(root, text='退出', command=exit_app, width=15, height=3)
exit_button.place(x=950, y=500)

# 创建图像预览区域
image_label = tk.Label(root)
image_label.place(x=100, y=50)

# 创建图像名称文本区域
name_text = tk.Text(root, height=1, width=4, font=('Arial', 108))
name_text.place(x=900, y=100)
name_text.configure(state="disabled")

# 运行主循环
root.resizable(False, False)
root.mainloop()
