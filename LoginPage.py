import csv
import os
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image

MAX_LOGIN_ATTEMPTS = 3

global tries , entered_username , entered_password

tries = 0

class LoginPage:

    def __init__(self, window):
        self.window = window
        self.window.geometry('1166x718')
        self.window.resizable(0, 0)
        self.window.state('zoomed')
        self.window.title('Login Page')
        
        photo = PhotoImage(file="images\\faceCopy.png")
        self.window.iconphoto(False,photo)

        # background image
        
        self.bg_frame = Image.open('images\\background1.png')
        photo = ImageTk.PhotoImage(self.bg_frame)
        self.bg_panel = Label(self.window, image=photo)
        self.bg_panel.image = photo
        self.bg_panel.pack(fill='both', expand='yes')
        
        # Login Frame 
        
        self.lgn_frame = Frame(self.window, bg='#040405', 
                               width=950, height=600)
        self.lgn_frame.place(x=200, y=70)

        self.txt = "WELCOME"
        self.heading = Label(self.lgn_frame, text=self.txt, 
                             font=('yu gothic ui', 25, "bold")
                             , bg="#040405",fg='white',bd=5,
                             relief=FLAT)
        self.heading.place(x=80, y=30, width=300, height=30)

        #  Left Side Image 
        self.side_image = Image.open('images\\vector.png')
        photo = ImageTk.PhotoImage(self.side_image)
        self.side_image_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.side_image_label.image = photo
        self.side_image_label.place(x=5, y=100)


        self.sign_in_image = Image.open('images\\hyy.png')
        photo = ImageTk.PhotoImage(self.sign_in_image)
        self.sign_in_image_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.sign_in_image_label.image = photo
        self.sign_in_image_label.place(x=620, y=130)


        self.sign_in_label = Label(self.lgn_frame, text="Sign In", bg="#040405", fg="white",
                                    font=("yu gothic ui", 17, "bold"))
        self.sign_in_label.place(x=650, y=240)


        self.username_label = Label(self.lgn_frame, text="Username", bg="#040405", fg="#4f4e4d",
                                    font=("yu gothic ui", 13, "bold"))
        self.username_label.place(x=550, y=300)
        self.username_entry = Entry(self.lgn_frame, highlightthickness=0, relief=FLAT, bg="#040405", fg="#6b6a69",
                                    font=("yu gothic ui ", 12, "bold"), insertbackground = '#6b6a69')
        self.username_entry.place(x=580, y=335, width=270)

        self.username_line = Canvas(self.lgn_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0)
        self.username_line.place(x=550, y=359)


        self.username_icon = Image.open('images\\username_icon.png')
        photo = ImageTk.PhotoImage(self.username_icon)
        self.username_icon_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.username_icon_label.image = photo
        self.username_icon_label.place(x=550, y=332)


        self.lgn_button = Image.open('images\\btn1.png')
        photo = ImageTk.PhotoImage(self.lgn_button)
        self.lgn_button_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.lgn_button_label.image = photo
        self.lgn_button_label.place(x=550, y=450)
        self.login = Button(self.lgn_button_label, text='LOGIN', font=("yu gothic ui", 13, "bold"), width=25, bd=0,
                            bg='#3047ff', cursor='hand2', activebackground='#3047ff', fg='white',command=self.loin)
        self.login.place(x=20, y=10)
      

        self.password_label = Label(self.lgn_frame, text="Password", bg="#040405", fg="#4f4e4d",
                                    font=("yu gothic ui", 13, "bold"))
        self.password_label.place(x=550, y=380)
        self.password_entry = Entry(self.lgn_frame, highlightthickness=0, relief=FLAT, bg="#040405", fg="#6b6a69",
                                    font=("yu gothic ui", 12, "bold"), show="*", insertbackground = '#6b6a69')
        self.password_entry.place(x=580, y=416, width=244)

        self.password_line = Canvas(self.lgn_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0)
        self.password_line.place(x=550, y=440)



        self.password_icon = Image.open('images\\password_icon.png')
        photo = ImageTk.PhotoImage(self.password_icon)
        self.password_icon_label = Label(self.lgn_frame, image=photo, bg='#040405')
        self.password_icon_label.image = photo
        self.password_icon_label.place(x=550, y=414)


        self.show_image = ImageTk.PhotoImage \
            (file='images\\show.png')

        self.hide_image = ImageTk.PhotoImage \
            (file='images\\hide.png')

        self.show_button = Button(self.lgn_frame, image=self.show_image, command=self.show, relief=FLAT,
                                  activebackground="white"
                                  , borderwidth=0, background="white", cursor="hand2")
        self.show_button.place(x=860, y=420)



    def show(self):
        self.hide_button = Button(self.lgn_frame, image=self.hide_image, command=self.hide, relief=FLAT,
                                  activebackground="white"
                                  , borderwidth=0, background="white", cursor="hand2")
        self.hide_button.place(x=860, y=420)
        self.password_entry.config(show='')

    def hide(self):
        self.show_button = Button(self.lgn_frame, image=self.show_image, command=self.show, relief=FLAT,
                                  activebackground="white"
                                  , borderwidth=0, background="white", cursor="hand2")
        self.show_button.place(x=860, y=420)
        self.password_entry.config(show='*')

    def read_user_data(self,file_path):
        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                return list(reader)
        except FileNotFoundError:
            messagebox.showerror("File Not Found", f"The file {file_path} was not found.")
            return []

    def authenticate(self,username, password):
        user_data = self.read_user_data("names.csv")
        
        for user in user_data:
            if user['username'] == username and user['password'] == password:
                return True
        return False

    def loin(self):

        global tries , entered_password , entered_username
        tries += 1 
        
        if tries < MAX_LOGIN_ATTEMPTS:
            entered_username = self.username_entry.get()
            entered_password = self.password_entry.get()

            if self.authenticate(entered_username, entered_password):
                messagebox.showinfo("Login Successful", f"Welcome, {entered_username}")
                with open("names.txt", 'w') as fie:
                    fie.write(entered_username)

                file = "pbl.py"
                self.window.destroy()
                os.system(f"python {file}")
                #exit()
            else:
                messagebox.showerror("Login Failed", f"Invalid username or password Tries Left {MAX_LOGIN_ATTEMPTS - tries}")
                self.username_entry.delete(0, END)
                self.password_entry.delete(0, END)
                self.username_entry.focus_set()
        else:
            messagebox.showerror("Too many attempts", "You have exceeded the maximum number of attempts.")
            self.window.destroy()


    def signin(self):
        pass



    
def page():
    window = Tk()
    LoginPage(window)
    window.mainloop()


if __name__ == '__main__':
    page()