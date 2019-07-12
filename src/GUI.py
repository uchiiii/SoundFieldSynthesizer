# codeing=utf-8
import tkinter as tk
import numpy as np

class GUI: 
    def __init__(self,r,r_c,Rint):
        self.root = tk.Tk()
        self.width = 600 #bit size of the room
        self.height = 400 #bit size of the room
        self.width_m = 5
        self.height_m = 3
        self.root.geometry("{}x{}".format(self.width,self.height))

        #self.Canvasを作る
        self.canvas = tk.Canvas(self.root, width=600, height=400, bg="white")
        self.canvas.place(x=0, y=0)

        self.r = r*self.height/self.height_m
        self.r[:,1] += self.width/2
        self.r[:,0] += self.height/10
        

        self.r_c = r_c*self.height/self.height_m
        self.r_c[:,1] += self.width/2
        self.r_c[:,0] += self.height/10

        self.Rint = Rint*self.height/self.height_m

        for i in range(self.r.shape[0]):
            self.canvas.create_oval(self.r[i,1]-5, self.r[i,0]-5, self.r[i,1]+5, self.r[i,0]+5, fill="red", width=0)

        for i in range(self.r_c.shape[0]):
            self.canvas.create_oval(self.r_c[i,1]-self.Rint[i], self.r_c[i,0]-self.Rint[i], self.r_c[i,1]+self.Rint[i], self.r_c[i,0]+self.Rint[i], fill="blue", width=0)

        #イベントを設定する
        self.canvas.bind("<Button-1>", self.click)

        self.root.mainloop()

    def click(self,e):
        #前の円を隠す
        #self.canvas.create_oval(x-30, y-30, x+30, y+30, fill="white", width=0)
        #クリックされた場所に描画する
        self.canvas.create_oval(e.x-20, e.y-20, e.x+20, e.y+20, fill="red", width=0)
        x = e.x
        y = e.y

if __name__=='__main__':
    NUM_L = 9 #the number of the used loudspeakers
    r = np.zeros((NUM_L,3))
    LENGTH_OF_TRIANGLE = 44.5 #44.5
    x_diff = LENGTH_OF_TRIANGLE/2.0*1.0e-2
    y_diff = LENGTH_OF_TRIANGLE*np.sqrt(3)/6*1.0e-2
    z_diff = (119-102.5)*1.0e-2
    #r[:,0] = np.arange(0,x_diff*NUM_L,x_diff)
    r[:,0] = np.linspace(0,x_diff*(NUM_L-1),NUM_L)
    for i in range(NUM_L):
        if i%2 == 0:
            r[i,1] = 0.0
            r[i,2] = -z_diff/2
        else:
            r[i,1] = -y_diff
            r[i,2] = z_diff/2
    Rint = np.array([0.2,0.05])
    is_silent = np.array([1, 0])
    r_c = np.array([[0.5 ,1.4 ,0],[1.5,1.4,0]]) #the center of target sphere 
    r_s = np.array([1.0,-2.0,0]) #the desired position of speaker
    gamma = np.array([5.0,1.0])
    gui = GUI(r,r_c,Rint)
	