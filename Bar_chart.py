#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
l=[[[10,20],[30,40],[50,60]],[[1,2],[4,5],[4,5]]]
a=np.array(l)
print(a)


# In[8]:


print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
print(a.itemsize)


# In[12]:


a=np.array([12,3.4,'f'],dtype='object')
print(a.dtype)


# In[16]:


a=np.arange(107,1,-10)
print(a)


# In[21]:


a=np.linspace(10,100,5,dtype=int)
print(a)


# In[31]:


a=np.full((2,2,3),'python')
print(a)


# In[37]:


a=np.identity(5,dtype=int)
print(a)


# In[40]:


a=np.empty((3,4,3))
print(a)


# In[44]:


a=np.random.randint(12,100,size=(2,2,9,3))
print(a)


# In[55]:


a=np.random.normal(2,4,size=(5,2))
print(a)


# In[63]:


a=np.random.randint(10,70,size=(4,3,5))
print(a)
b=np.random.shuffle(a)
print(a)


# In[72]:


a=np.random.randint(1,101,size=(2,3,4,5,5,3,5,5))
for x in np.nditer(a,flags=['buffered'],op_dtypes=['float']):
    print(x)


# In[74]:


a=np.random.randint(1,101,size=(2,3,2,3,4))
for pos,element in np.ndenumerate(a):
    print("element {} present in index{}".format(pos,element))
    


# In[77]:


a=np.arange(1,11)
print(a[54])


# In[85]:


a=np.random.randint(1,101,size=(2,2,2,4))
print(a)
print(a[1][1][1][2])


# In[86]:


a=np.arange(1,101)
print(a+2)
print(a-2)
print(a*2)
print(a/2)
print(a//2)
print(a/2)
print(a**2)


# In[87]:


a=np.arange(0,51)
print(a/0)


# In[97]:


a=np.random.randint(1,101,size=(2,6,7))
b=np.random.randint(1,101,size=(2,6,7))

print(np.power(a,b))


# In[100]:


a=np.random.randint(1,101,size=(3,1))
b=np.random.randint(1,101,size=(1,3))
print(a+b)


# In[105]:


a=np.arange(24).reshape(12,2)
b=np.reshape(a,(2,-34,4))
print(a)


# In[106]:


a=np.random.randint(1,101,size=(12))
b=a.reshape(2,-90)
print(b)


# In[107]:


a[0]=99
print(a)
print(b)


# In[ ]:


import numpy as np
a=np.arange(24).reshape()


# In[2]:


import numpy as np
a=np.array([10,20,30])
b=np.array([50,60,70,80])
c=np.vstack((a,b))
print(c)


# In[12]:


a=np.arange(1,25).reshape(2,3,4)
print(a)
b=np.arange(25,49).reshape(2,3,4)
print(b)
c=np.hstack((a,b))
print(c)


# In[10]:


a=np.array([10,20,30,40])
b=np.array([50,60,70,80,90,100])
c=np.hstack((a,b))
print(c)


# In[11]:


a=np.arange(1,7).reshape(3,2)
b=np.arange(7,16).reshape(3,3)
c=np.hstack((a,b))
print(c)


# In[3]:


import numpy as np
a=np.arange(1,25).reshape(2,3,4)
#print(a)
b=np.arange(25,49).reshape(2,3,4)
#print(b)
c=np.hstack((a,b))
print(c)


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
plt.plot(a,b,'o--y',label='identity')
plt.plot(a,c,'o--c',label='square')
plt.plot(a,a,'o--g',label='cube')
plt.legend(loc=(0,1.1),ncol=2,title='3 common color')


# In[72]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
plt.figure(num=1,figsize=(10,5),facecolor='green')
line1,line2,line3=plt.plot(a,b,'o-y',a,c,'^:c',a,a,'o--m')
plt.grid(which='major',lw=3,color='g',axis='both')
plt.minorticks_on()
plt.grid(which='minor',color='r')

plt.legend([line1,line2,line3],['identity','square','cubes'],ncol=1,loc=(1,1.1),title='demo')


# In[7]:


import matplotlib.pyplot as plt
year=[2018,2019,2020,2021,2022]
sales=[12000,40000,2500,30000,10000]
c=['r','b','lime','y','orange']
w=[0.3,0.4,0.2,0.8,0.9]
plt.bar(year,sales,color=c,width=w,align='edge')


# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import csv
name=np.array([],dtype='str')
marks=np.array([],dtype='int')
f=open('student.csv','r')
r=csv.reader(f)
h=next(r)
for i in r:
    name=np.append(name,i[0])
    marks=np.append(marks,i[1])
print(name)
print(marks)
plt.barh(name,marks)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
x=np.random.rand(10000)
plt.hist(x,bins=15)


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
fig=plt.figure(num=1,figsize=(10,8))
ax1=fig.add_axes([0.3,0.3,0.2,0.2])
ax1.plot(a,a,'o-y',a,b,'o-c',a,c,lw=3,ms=10,mfc='orange')
ax2=fig.add_axes([0.6,0.1,0.4,0.3])
ax1.plot(a,a,'o-y',a,b,'o-c',a,c,lw=3,ms=10,mfc='orange')


# In[44]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
fig,ax=plt.subplots(3,2)

ax1.plot(a,b,'o-g',ms=10,mfc='lime')
ax1.set(xlabel='N',ylabel='SQUARE OF N',title='SQUARE FUNCTION')


ax2.plot(a,c,'o-g',ms=10,mfc='lime')
ax2.set(xlabel='N',ylabel='CUBIC OF N',title='CUBIC FUNCTION')


ax3.plot(a,b,'o-g',ms=10,mfc='lime')
ax3.set(xlabel='N',ylabel='SQUARE OF N',title='SQUARE FUNCTION')


ax4.plot(a,c,'o-g',ms=10,mfc='lime')
ax4.set(xlabel='N',ylabel='CUBIC OF N',title='CUBIC FUNCTION')


ax5.plot(a,b,'o-g',ms=10,mfc='lime')
ax5.set(xlabel='N',ylabel='SQUARE OF N',title='SQUARE FUNCTION')


ax6.plot(a,c,'o-g',ms=10,mfc='lime')
ax6.set(xlabel='N',ylabel='CUBIC OF N',title='CUBIC FUNCTION')
plt.savefig('C:/Users/anjan Roy/OneDrive/desktop/Matplotlib/pic4.jpeg')
plt.tight_layout()


# In[42]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
fig,ax=plt.subplots(2,3)
print(fig)
print(ax)


# In[48]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3)

ax1.plot(a,b,'o-g',ms=10,mfc='lime')
ax1.set(xlabel='N',ylabel='SQUARE OF N',title='SQUARE FUNCTION')


ax2.plot(a,c,'o-g',ms=10,mfc='lime')
ax2.set(xlabel='N',ylabel='CUBIC OF N',title='CUBIC FUNCTION')


ax3.plot(a,b,'o-g',ms=10,mfc='lime')
ax3.set(xlabel='N',ylabel='SQUARE OF N',title='SQUARE FUNCTION')


ax4.plot(a,c,'o-g',ms=10,mfc='lime')
ax4.set(xlabel='N',ylabel='CUBIC OF N',title='CUBIC FUNCTION')


ax5.plot(a,b,'o-g',ms=10,mfc='lime')
ax5.set(xlabel='N',ylabel='SQUARE OF N',title='SQUARE FUNCTION')


ax6.plot(a,c,'o-g',ms=10,mfc='lime')
ax6.set(xlabel='N',ylabel='CUBIC OF N',title='CUBIC FUNCTION')
plt.savefig('C:/Users/anjan Roy/OneDrive/desktop/Matplotlib/pic4.jpeg')
plt.tight_layout()


# In[55]:


import matplotlib.pyplot as plt
import numpy as np
a=np.arange(1,11)
b=a**2
c=a**3
fig=plt.figure(figsize=(10,8))
ax1=plt.subplot(2,2,1)
ax1.plot(a,b)
ax2=plt.subplot(2,2,2)
ax2.plot(a,b)
ax3=plt.subplot(2,2,3)
ax3.plot(a,b)
ax4=plt.subplot(2,2,4)
ax4.plot(a,b)


# In[17]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
plt.figure(figsize=(12,8))
m=Basemap()
m.drawcoastlines()
#m.drawrivers()
m.drawlsmask(land_color='g',ocean_color='y',lakes=True)
m.drawparallels(np.arange(-90,90,10),labels=[True,True,True,True])
m.drawmeridians(np.arange(-180,180,30),labels=[True,True,True,True])


# In[22]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
m=Basemap(projection='mill',resolution='c')
#m.bluemarble()
m.etopo()


# In[5]:


import pandas as pd
name=['virat','dhoni','sachin']
runs=[23000,45000,65000]
s=pd.Series(data=runs,index=name,name='runs of players')
print(s)


# In[8]:


import pandas as pd
s=pd.read_csv('student_details.csv',usecols=['name','marks'],squeeze=True)
print(type(s))


# In[6]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df['salary'])
print('-'*50)
print(df['salary'].rank(ascending=True))


# In[22]:


import pandas as pd
df=pd.read_csv('student_details.csv')
df.dropna(how='all',inplace=True)
df['salary']=df['salary'].fillna(0).astype('int')
df['salary rank']=df['salary'].rank().astype('int')
df.sort_values(by='salary rank',inplace=True)
print(df)


# In[58]:


import pandas as pd
df=pd.read_csv('student_details.csv',index_col='name')
print(df)
print('-'*50)
print(df.loc[['jinny','binny'],['salary','marks']])


# In[17]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df)
print('-'*50)


# In[ ]:





# In[73]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df)
print('-'*50)
df.iloc[[1,4,2],3]=100
def update_address(row):
    if row['Address']=='kolkata':
        return 'madhyamgram'
    else:
        return row['Address']
df['Address']=df.apply(update_address,axis=1)
df


# In[12]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df)
print('-'*50)
def update_salary(row):
    if row['salary']>=64000:
        return 'director'
    elif row['salary']>=40000:
        return 'precident'
    else:
        return 'consultant'
df['designation']=df.apply(update_salary,axis=1)
df


# In[37]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df)
print('-'*50)
c=df['names']=df['names'].str.strip()
df


# In[ ]:





# In[7]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df)
print('-*30')
df['salary']=df['salary'].str.replace(r'[$,]','',regex=True).astype('float')
df


# In[17]:


import pandas as pd
df=pd.read_csv('student_details.csv')
print(df)
print('-'*30)
c=df['name'].str.contains(r'[^j]',regex=True)
df[c]


# In[21]:


import re
matcher=re.finditer('[^j]','sdvkjab')
for m in matcher:
    print(m.start())


# In[15]:


import pandas as pd
df=pd.read_csv('new_batch_1.csv')
df.set_index(keys=['date','subject'],inplace=True)
print(df)
print('='*30)
print(df.loc[slice('04-01-2023','06-01-2023'),('faculty','timimg')])


# # Horizental Bar Chart

# In[16]:


import matplotlib.pyplot as plt
import numpy as np
names=np.array(['rohit','dhoni','anjan','ranbir'])
gold=np.array([34,60,50,100])
silver=np.array([90,50,60,80])
bronze=np.array([100,40,60,90])
plt.barh(names,gold,color='g')
plt.barh(names,silver,color='y',left=gold)
plt.barh(names,bronze,color='c',left=gold+silver)
for i in range(names.size):
    plt.text(gold[i]/2,names[i],gold[i])
    plt.text(gold[i]+silver[i]/2,names[i],silver[i])
    plt.text(gold[i]+silver[i]+bronze[i]/2,names[i],bronze[i])

