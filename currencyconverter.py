import tkinter
from PIL import Image , ImageTk
from tkinter import ttk
def graph():
   main= tkinter.Toplevel(root)
# def conversion():


root=tkinter.Tk()
root.title("Currency Converter")
root.config(bg="light blue")
root.geometry("300x400")
imagepath=r"/Users/srishtigupta/Desktop/pythonproject/currencypic.jpg"
image=Image.open(imagepath)
image_convert=ImageTk.PhotoImage(image)
image_label=tkinter.Label(root,image=image_convert)
image_label.pack(pady=10)
list1=["IND","PAK","ENG","AUS"]
label1=tkinter.Label(text="from currency")
label1.pack()
combobox1=ttk.Combobox(root,value=list1)
combobox1.pack()
list2=["IND","PAK","ENG","AUS"]
label2=tkinter.Label(text="to currency")
label2.pack()
combobox2=ttk.Combobox(root,value=list2)
combobox2.pack()
frame=tkinter.Frame(root)
frame.pack()
button=tkinter.Button(root,text="convert",bg="pink")#,command=conversion)
button.pack(pady=30)
button2=tkinter.Button(root,text="graph",bg="pink",activebackground="green", activeforeground="yellow" , command=graph)
button2.pack(pady=30)
root.mainloop()
