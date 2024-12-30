@echo off
rem Chạy chương trình C# ẩn
D:\TDDDDD\nircmd-x64\nircmd exec hide "D:\Honda_PlusVN\Python\Python_Project\YOLOX\C#_Code\bin\Debug\net6.0\DACN.exe"

rem Kích hoạt môi trường Python và chạy script Python ẩn
D:\TDDDDD\nircmd-x64\nircmd exec hide cmd /c "cd /d D:\Honda_PlusVN\Python\Python_Project && call .venv\Scripts\activate && cd YOLOX && python camlaptop.py"

rem Thoát khỏi command prompt
exit
