# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:33:01 2021

@author: me1mwhx
"""

import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


 #opens dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
file=file_path
#either open dialogue using code above or use below code to set path 

#path = r"C:\Users\me1mwhx\Documents\TRENT\Task 2- Test Development\Contact conditions\Friction Rig\prelim#\05- Dynamic Tests"
#file_path = glob.glob(path + "/*.txt")

#file=file_path[0]

df_1 = pd.read_csv(file)
df_1.columns = ['Fx', 'Fy','Fz','Mx','My','Mz']

# file=file_path[1]
# df_2 = pd.read_csv(file)
# df_2.columns = ['Fx', 'Fy','Fz','Mx','My','Mz']

x=np.arange (0, 2999, 1)
t=x/50
plt.figure()
#plt.plot(t,df_1.Fx)
plt.plot(t,df_1.Fy)
plt.plot(t,df_1.Fz)
# plt.plot(t,df_2.Fy)
# plt.plot(t,df_2.Fz)
plt.legend(('Fy', 'Fz'),loc="upper right")
plt.ylabel('Force')
plt.xlabel('Time')
a=(df_1["Fz"].max())
plt.axis([0, 5, 0,a+1])
plt.show()


a= df_1.Fz
b=df_1.Fy
CoF_1=df_1.Fy/df_1.Fz
CoF_1[0:1100]=0
# CoF_2=df_2.Fy/df_2.Fz
plt.figure()
plt.plot(t,CoF_1)
# plt.plot(t,CoF_2)
plt.ylabel('Coefficient of Friction')
plt.xlabel('Time')
#plt.legend(('1N','10N'),loc="upper right")
plt.axis([0, 60, 0, 1])
plt.show()
