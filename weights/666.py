import PySimpleGUI as sg
layout=[
[sg.T('小帽',key='-Text-'),sg.B('点赞')]
]

# 3.创建窗口
window = sg.Window('Python GUI', layout)

# 4.事件循环
while True:
    event, values = window.read()   # 窗口的读取，有两个返回值（1.事件，2.值）
    if event == None:   # 窗口关闭事件
        break
    if event == '点赞':
        window['-Text-'].update(
            value='谢谢支持！',                 # str 更新文本
            background_color = 'white',    # 更新文本背景颜色
            text_color = 'black',          # 更新文本颜色
            font = ('黑体', 30),                # 更新字体的名称或者大小
            visible = None              # 更新元素的可见状态
        )

# 5.关闭窗口
window.close()
