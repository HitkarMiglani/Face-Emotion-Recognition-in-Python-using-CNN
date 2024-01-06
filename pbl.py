import csv
import os
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image


class UserPage:

    def __init__(self, window):
        self.window = window
        self.window.geometry('1166x718')
        self.window.resizable(0, 0)
        self.window.state('zoomed')
        self.window.title('Face Emotion Recognition')

        photo = PhotoImage(file="images\\faceCopy.png")
        self.window.iconphoto(False,photo)


        self.bg_frame = Image.open('images\\bg.png')
        photo = ImageTk.PhotoImage(self.bg_frame)
        self.bg_panel = Label(self.window, image=photo)
        self.bg_panel.image = photo
        self.bg_panel.pack(fill='both', expand='yes')
        

        self.lgn_frame = Frame(self.window, bg='#040405', width=950, height=600)
        self.lgn_frame.place(x=200, y=70)


        with open("Names.txt") as f:
            user = f.read()

        self.txt = f"Welcome {user} "
        self.heading = Label(self.lgn_frame, text=self.txt, font=('yu gothic ui', 25, "bold"), bg="#040405",
                             fg='white',
                             bd=5,
                             relief=FLAT)
        self.heading.place(x=80, y=30, width=300, height=30)


        self.side_image = Image.open('images\\fce.png')
        photo = ImageTk.PhotoImage(self.side_image)
        self.side_image_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.side_image_label.image = photo
        self.side_image_label.place(x=5, y=100)



        self.sign_in_image = Image.open('images\\fa.png')
        photo = ImageTk.PhotoImage(self.sign_in_image)
        self.sign_in_image_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.sign_in_image_label.image = photo
        self.sign_in_image_label.place(x=620, y=100)




        self.lgn_button = Image.open('images\\btn1.png')
        photo = ImageTk.PhotoImage(self.lgn_button)

        self.lgn_button_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.lgn_button_label.image = photo
        self.lgn_button_label.place(x=550, y=280)
        self.login = Button(self.lgn_button_label, text='Sample', font=("yu gothic ui", 13, "bold"), width=25, bd=0,
                            bg='#3047ff', cursor='hand2', activebackground='#3047ff', fg='white',command=self.sample)
        self.login.place(x=20, y=10)
      


    
        self.real_button_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.real_button_label.image = photo
        self.real_button_label.place(x=550, y=350)
        self.real = Button(self.real_button_label, text='Real Time Detection', font=("yu gothic ui", 13, "bold"),       
                           width=25, bd=0,bg='#3047ff', cursor='hand2', activebackground='#3047ff', fg='white',command=self.real)
        self.real.place(x=20, y=10)



        self.user_button_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.user_button_label.image = photo
        self.user_button_label.place(x=550, y=420)
        self.userr = Button(self.user_button_label, text='Add User', font=("yu gothic ui", 13, "bold"), width=25, bd=0,
                            bg='#3047ff', cursor='hand2', activebackground='#3047ff', fg='white',command=self.User)
        self.userr.place(x=20, y=10)


        self.Quit_button_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.Quit_button_label.image = photo
        self.Quit_button_label.place(x=550, y=490)
        self.Quit = Button(self.Quit_button_label, text='Quit Application', font=("yu gothic ui", 13, "bold"), 
                           width=25, bd=0, bg='#3047ff', cursor='hand2', activebackground='#3047ff', 
                           fg='white',command=self.exit)
        self.Quit.place(x=20, y=10)


    def sample(self):
        file = "sample.py"

        os.system(f"python {file}")

    def real(self):
        file = "realtimedetection.py"

        os.system(f"python {file}")

    def User(self):
        def detail():
            user = username.get()
            password = passwor.get()
            if (username == "" or password == "") :
                messagebox.showerror("Error", "Please fill all fields")
                root.destroy()
            else:
                with open("Names.csv","a") as f:
                    writer = csv.writer(f)
                    writer.writerow([user,password])
                    messagebox.showinfo('Success', 'User Added Successfully!')
                
                root.destroy()


        root = Tk()
        root.geometry("250x300")
        root.title("Add New User")
        root.config(bg= "lightblue")

        Username = Label(root,text="New Username: " , bg="lightblue")
        Password = Label(root, text="Password : " , bg="lightblue")
        username = Entry(root)
        passwor = Entry(root, show="*")
        submit = Button(root, text="Submit", command=detail)
        Username.grid(row=2, column=5)
        Password.grid(row=3,column=5)
        username.grid(row=2, column=7)
        passwor.grid(row=3, column=7)
        submit.grid(pady=5,row=6,column=7, columnspan=3)

        root.mainloop()

    def exit(self):
        if messagebox.askquestion("Confirmation","Are you Sure")=='yes':
            self.window.destroy()
        else:
            pass




def page():
    window = Tk()
    UserPage(window)
    window.mainloop()


if __name__ == '__main__':
    page()