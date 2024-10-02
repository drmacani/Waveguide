
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pixels=np.load(r"C:\Users\me1mwhx\Documents\TRENT\Task 2- Test Development\Contact conditions\Friction Rig\Phase 1\T07\pixels.npy")
#area=pixels*0.0002741
file=r"C:\Users\me1mwhx\Documents\TRENT\Task 2- Test Development\Contact conditions\Friction Rig\Phase 1\T07\T07.txt"
df_1 = pd.read_csv(file)
df_1.columns = ['Fx', 'Fy','Fz','Mx','My','Mz']
length=pixels.shape[0]

#pixels=np.load(r"C:\Users\me1mwhx\Documents\TRENT\Task 2- Test Development\Contact conditions\Friction Rig\prelim#\05- Dynamic Tests\Initial\pixel_num.npy")
area=pixels*0.0000998
x=np.arange (0, length, 1)
t=x/20
t=np.around(t,3)
plt.figure()
plt.plot(t,area)
plt.xlabel("Time (s)")
plt.ylabel("Contact Area (mm²)")
plt.show()


xf=np.arange (0, 2999, 1)
tf=xf/50
plt.figure()
#plt.plot(t,df_1.Fy)
plt.plot(tf,df_1.Fz)
#plt.legend(('Fy', 'Fz'),loc="upper right")
plt.ylabel('Force')
plt.xlabel('Time')
a=(df_1["Fz"].max())
plt.axis([0, 60, 0,a+1])
plt.show()

sframe=162
fframe=588
# sindex=int((sframe/30)*50)
# findex=int((fframe/30)*50)

sindex=470
findex=int(sindex+(fframe/30)*50)

#area=area[32:555]

force=df_1.Fz.values
force=force[sindex:findex]   
tf2=tf[sindex:findex]
tf2=tf2-tf2[0]
tf2=np.around(tf2,3)

plt.figure()
plt.plot(tf2,force)
plt.plot(t,area)
plt.show()

timeindex=np.arange(0,16.3,0.1)
timeindex=np.around(timeindex,3)
count=0
f2=np.zeros(timeindex.shape,dtype=float)
a2=np.zeros(timeindex.shape,dtype=float)
for index,ti in enumerate(timeindex):
    b=np.where(ti==tf2)
    f2[count]=force[b[0]]
    c=np.where(ti==t)
    a2[count]=area[c[0]]
    count=count+1
    
plt.figure()

plt.plot(f2,a2) 
plt.xlabel("Force (N))")
plt.ylabel("Contact Area (mm²)")
plt.show()


# x=np.zeros(523,dtype=float)
# count=0
# for index, element in enumerate(force):
#     if index % 2 == 0:
#         x[count]=element
#         count=count+1
        
# plt.figure()
# plt.plot(x,area)
# plt.xlabel("Force (N))")
# plt.ylabel("Contact Area (mm²)")
# plt.show()


