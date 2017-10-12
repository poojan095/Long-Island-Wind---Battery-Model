
# coding: utf-8

# In[2]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the Census dataset
data = pd.read_csv("new.csv")

# Success - Display the first record
display(data.head(n=10))


# In[3]:

n_records = len(data)
print "Total number of records: {}".format(n_records)


# # Plotting data

# In[4]:

import matplotlib.pyplot as plt
import csv
import datetime


Y = data['Area2-Bridgehampton&W/OHither Hills'];
date = data['Date New']
fig, ax = plt.subplots()

plt.plot_date(date[3500:8000],Y[3500:8000],markersize=.1,linewidth=1, linestyle="-")
plt.gcf().autofmt_xdate()
ax.set_xlim([datetime.date(2013, 6, 20), datetime.date(2013, 8, 30)])
ax.set_ylim([15, 90])
plt.gcf().autofmt_xdate()

#plt.rcParams["figure.figsize"] = [16,9]

plt.show()



# # Month peak average

# In[5]:

def month_peak_avg(month,n_days,year):

    mon_v1 = []
    mon_v2 = []
    mon_v3 = []
    if year == 2013:
        if month == 1:
            start = 0
        if month == 2:
            start = 30
        if month == 3:
            start = 58
        if month == 4:
            start = 89
        if month == 5:
            start = 119
        if month == 6:
            start = 150
        if month == 7:
            start = 180
        if month == 8:
            start = 211
        if month == 9:
            start = 242
        if month == 10:
            start = 272
        if month == 11:
            start = 303
        if month == 12:
            start = 333
            
     
    if year == 2014:
        if month == 1:
            start = 364 + 0
        if month == 2:
            start = 364 + 30
        if month == 3:
            start = 364 + 58
        if month == 4:
            start = 364 + 89
        if month == 5:
            start = 364 + 119
        if month == 6:
            start = 364 + 150
        if month == 7:
            start = 364 + 180
        if month == 8:
            start = 364 + 211
        if month == 9:
            start = 364 + 242
        if month == 10:
            start = 364 + 272
        if month == 11:
            start = 364 + 303
        if month == 12:
            start = 364 + 333
            
     
    if year == 2015:
        if month == 1:
            start = 728 + 0
        if month == 2:
            start = 728 + 30
        if month == 3:
            start = 728 + 58
        if month == 4:
            start = 728 + 89
        if month == 5:
            start = 728 + 119
        if month == 6:
            start = 728 + 150
        if month == 7:
            start = 728 + 180
        if month == 8:
            start = 728 + 211
        if month == 9:
            start = 728 + 242
        if month == 10:
            start = 728 + 272
        if month == 11:
            start = 728 + 303
        if month == 12:
            start = 728 + 333
            
  



    for i in range(start, start + n_days):
        t = i*24
        high1 = 0
        high2 = 0
        high3 = 0
        
        for j in range(t,t+24):

            if data['Area1 Amagansett'][j] > high1:
                high1 = data['Area1 Amagansett'][j]
            if data['Area2-Bridgehampton&W/OHither Hills'][j] > high2:
                high2 = data['Area2-Bridgehampton&W/OHither Hills'][j]
            if data['Area3-Canal&W/OE.Hampton/Buell'][j] > high3:
                high3 = data['Area3-Canal&W/OE.Hampton/Buell'][j]

        mon_v1.append(high1)
        mon_v2.append(high2)
        mon_v3.append(high3)

    avg1 = sum(mon_v1)/len(mon_v1)
    avg2 = sum(mon_v2)/len(mon_v2)
    avg3 = sum(mon_v3)/len(mon_v3)

    return avg1, avg2, avg3


# In[6]:

import matplotlib.patches as mpatches
import datetime as dt

average = []
average.append(month_peak_avg(1,31,2013))
average.append(month_peak_avg(2,28,2013))
average.append(month_peak_avg(3,31,2013))
average.append(month_peak_avg(4,30,2013))
average.append(month_peak_avg(5,31,2013))
average.append(month_peak_avg(6,30,2013))
average.append(month_peak_avg(7,31,2013))
average.append(month_peak_avg(8,31,2013))
average.append(month_peak_avg(9,30,2013))
average.append(month_peak_avg(10,31,2013))
average.append(month_peak_avg(11,30,2013))
average.append(month_peak_avg(12,31,2013))

average.append(month_peak_avg(1,31,2014))
average.append(month_peak_avg(2,28,2014))
average.append(month_peak_avg(3,31,2014))
average.append(month_peak_avg(4,30,2014))
average.append(month_peak_avg(5,31,2014))
average.append(month_peak_avg(6,30,2014))
average.append(month_peak_avg(7,31,2014))
average.append(month_peak_avg(8,31,2014))
average.append(month_peak_avg(9,30,2014))
average.append(month_peak_avg(10,31,2014))
average.append(month_peak_avg(11,30,2014))
average.append(month_peak_avg(12,31,2014))

average.append(month_peak_avg(1,31,2015))
average.append(month_peak_avg(2,28,2015))
average.append(month_peak_avg(3,31,2015))
average.append(month_peak_avg(4,30,2015))
average.append(month_peak_avg(5,31,2015))
average.append(month_peak_avg(6,30,2015))
average.append(month_peak_avg(7,31,2015))
average.append(month_peak_avg(8,31,2015))
average.append(month_peak_avg(9,30,2015))
average.append(month_peak_avg(10,31,2015))
average.append(month_peak_avg(11,30,2015))
#average.append(month_peak_avg(12,31,2015))


#AMagansett Area 1
y1 = []

for j in range(0,35):
    y1.append(average[j][0])
    
x1 = []
for year in range(2013, 2016):
    for month in range(1, 13):
        x1.append(dt.datetime(year=year, month=month, day=1))
    
x1.pop(35)
print y1
plt.plot_date(x1,y1,color='b',markersize=.1,linewidth=1.5, linestyle="-")

blue_patch = mpatches.Patch(color='blue', label='Area 1 - Amagansett')
plt.legend(handles=[blue_patch])

plt.title('Average Peaks each month for 2013')

plt.xlabel("Year - Month")
plt.ylabel("Consumption")
plt.show()


#Area 2 and Area 3
y2 = []

for j in range(0,35):
    y2.append(average[j][1])
x2 = x1


y3 = []

for j in range(0,35):
    y3.append(average[j][2])
x3 = x1

plt.plot(x3,y3,color='r')
plt.plot(x2,y2,color='g')

plt.xlabel("Year - Month")
plt.ylabel("Consumption")

red_patch = mpatches.Patch(color='red', label='Area 3 - East Hampton')
green_patch = mpatches.Patch(color='green', label='Area 2 - Bridgehampton')
plt.legend(handles=[green_patch, red_patch])
plt.gcf().autofmt_xdate()

plt.title('Average Peaks each month for 2013')
plt.show()


# # Monthly average

# In[7]:

def monthly_avg(month,n_days,year):

    mon_total_1 = []
    mon_total_2 = []
    mon_total_3 = []

   
    if year == 2013:
        if month == 1:
            start = 0
        if month == 2:
            start = 30
        if month == 3:
            start = 58
        if month == 4:
            start = 89
        if month == 5:
            start = 119
        if month == 6:
            start = 150
        if month == 7:
            start = 180
        if month == 8:
            start = 211
        if month == 9:
            start = 242
        if month == 10:
            start = 272
        if month == 11:
            start = 303
        if month == 12:
            start = 333
            
     
    if year == 2014:
        if month == 1:
            start = 364 + 0
        if month == 2:
            start = 364 + 30
        if month == 3:
            start = 364 + 58
        if month == 4:
            start = 364 + 89
        if month == 5:
            start = 364 + 119
        if month == 6:
            start = 364 + 150
        if month == 7:
            start = 364 + 180
        if month == 8:
            start = 364 + 211
        if month == 9:
            start = 364 + 242
        if month == 10:
            start = 364 + 272
        if month == 11:
            start = 364 + 303
        if month == 12:
            start = 364 + 333
            
     
    if year == 2015:
        if month == 1:
            start = 728 + 0
        if month == 2:
            start = 728 + 30
        if month == 3:
            start = 728 + 58
        if month == 4:
            start = 728 + 89
        if month == 5:
            start = 728 + 119
        if month == 6:
            start = 728 + 150
        if month == 7:
            start = 728 + 180
        if month == 8:
            start = 728 + 211
        if month == 9:
            start = 728 + 242
        if month == 10:
            start = 728 + 272
        if month == 11:
            start = 728 + 303
        if month == 12:
            start = 728 + 333
            
            
  



    for i in range(start, start + n_days):
        t = i*24
        
        for j in range(t,t+24):

            mon_total_1.append(data['Area1 Amagansett'][j])
            mon_total_2.append(data['Area2-Bridgehampton&W/OHither Hills'][j])
            mon_total_3.append(data['Area3-Canal&W/OE.Hampton/Buell'][j])

    avg1 = sum(mon_total_1)/len(mon_total_1)
    avg2 = sum(mon_total_2)/len(mon_total_2)
    avg3 = sum(mon_total_3)/len(mon_total_3)
 
    return avg1, avg2, avg3


# In[8]:

average2 = []
average2.append(monthly_avg(1,31,2013))
average2.append(monthly_avg(2,28,2013))
average2.append(monthly_avg(3,31,2013))
average2.append(monthly_avg(4,30,2013))
average2.append(monthly_avg(5,31,2013))
average2.append(monthly_avg(6,30,2013))
average2.append(monthly_avg(7,31,2013))
average2.append(monthly_avg(8,31,2013))
average2.append(monthly_avg(9,30,2013))
average2.append(monthly_avg(10,31,2013))
average2.append(monthly_avg(11,30,2013))
average2.append(monthly_avg(12,31,2013))

average2.append(monthly_avg(1,31,2014))
average2.append(monthly_avg(2,28,2014))
average2.append(monthly_avg(3,31,2014))
average2.append(monthly_avg(4,30,2014))
average2.append(monthly_avg(5,31,2014))
average2.append(monthly_avg(6,30,2014))
average2.append(monthly_avg(7,31,2014))
average2.append(monthly_avg(8,31,2014))
average2.append(monthly_avg(9,30,2014))
average2.append(monthly_avg(10,31,2014))
average2.append(monthly_avg(11,30,2014))
average2.append(monthly_avg(12,31,2014))

average2.append(monthly_avg(1,31,2015))
average2.append(monthly_avg(2,28,2015))
average2.append(monthly_avg(3,31,2015))
average2.append(monthly_avg(4,30,2015))
average2.append(monthly_avg(5,31,2015))
average2.append(monthly_avg(6,30,2015))
average2.append(monthly_avg(7,31,2015))
average2.append(monthly_avg(8,31,2015))
average2.append(monthly_avg(9,30,2015))
average2.append(monthly_avg(10,31,2015))
average2.append(monthly_avg(11,30,2015))
#average2.append(monthly_avg(12,31,2015))


#Amagansett Area 1
y1 = []
Y1 = []
for j in range(0,35):
    y1.append(average2[j][0])
for j in range(0,35):
    Y1.append(average[j][0])
    

    
x1 = []
for year in range(2013, 2016):
    for month in range(1, 13):
        x1.append(dt.datetime(year=year, month=month, day=1))
    
x1.pop(35)

plt.plot_date(x1,y1,color='b',markersize=.1,linewidth=1.5, linestyle="-")
plt.plot_date(x1,Y1,color='g',markersize=.1,linewidth=1.5, linestyle="-")


blue_patch = mpatches.Patch(color='blue', label='Area 1 - peak average')
green_patch = mpatches.Patch(color='green', label='Area 1 - usage average')

plt.legend(handles=[blue_patch, green_patch])

plt.title('Area 1 - Average electricity consumption each month for 2013')

plt.xlabel("Year - Month")
plt.ylabel("Consumption")
plt.show()


#Area 2 and Area 3
y2 = []
Y2 = []
for j in range(0,35):
    y2.append(average2[j][1])
    
for j in range(0,35):
    Y2.append(average[j][1])
x2 = x1

plt.plot(x2,y2,color='g')
plt.plot(x2,Y2,color='y')
green_patch = mpatches.Patch(color='green', label='Area 2 - peak average')
yellow_patch = mpatches.Patch(color='yellow', label='Area 2 - usage average')
plt.legend(handles=[green_patch, yellow_patch])
plt.xlabel("Year - Month")

plt.title('Area 2 - Average electricity consumption each month for 2013')
plt.show()

y3 = []
Y3 = []
for j in range(0,35):
    y3.append(average2[j][2])

for j in range(0,35):
    Y3.append(average[j][2])

    
x3 = x1

plt.plot(x3,y3,color='r')
plt.plot(x3,Y3,color='k')

plt.xlabel("Year - Month")
plt.ylabel("Consumption")

red_patch = mpatches.Patch(color='red', label='Area 3 - peak average')
black_patch = mpatches.Patch(color='black', label='Area 3 - usage average')


plt.legend(handles=[red_patch, black_patch])

plt.title('Average electricity consumption each month for 2013')
plt.show()


# # Daily Averages

# In[9]:

daily_average_1_2013 = np.zeros((7,24))
x = 1


# In[10]:

for i in range(0,31):
        t = i*24
        
        for j in range(t,t+24):
            daily_average_1_2013[x][j-t] = daily_average_1_2013[x][j-t] + data['Area1 Amagansett'][j]
        if x == 6:
            x = 0
        elif x < 6:
            x = x + 1

for i in range(0,24):
    daily_average_1_2013[0][i] = daily_average_1_2013[0][i]/4
    daily_average_1_2013[1][i] = daily_average_1_2013[0][i]/4
    daily_average_1_2013[2][i] = daily_average_1_2013[0][i]/5
    daily_average_1_2013[3][i] = daily_average_1_2013[0][i]/5
    daily_average_1_2013[4][i] = daily_average_1_2013[0][i]/5
    daily_average_1_2013[5][i] = daily_average_1_2013[0][i]/4
    daily_average_1_2013[6][i] = daily_average_1_2013[0][i]/4
    


# In[11]:

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]


# In[12]:

plt.plot(x,daily_average_1_2013[0][:])
plt.show()


# # Wind Data

# In[13]:

temp = pd.read_csv("wind_13-16.csv")
display(temp.head(n=10))


# In[14]:

temp1 = temp.drop('off.J1',axis = 1)
data_wind = temp1.drop('k - long island', axis = 1)


# In[15]:

display(data_wind.head(n=10))


# In[16]:

n_records = len(data_wind)
print "Total number of records: {}".format(n_records)


# In[17]:

Y = data_wind['off.K1'];
fig, ax = plt.subplots()
date = data_wind['Date New']
plt.plot_date(date,Y,markersize=.1,linewidth=.4, linestyle="-")
plt.gcf().autofmt_xdate()
ax.set_xlim([datetime.date(2014, 7, 1), datetime.date(2014, 7, 31)])
ax.set_ylim([0, 1])
#plt.rcParams["figure.figsize"] = [16,9]

plt.show()

print "\n\nJan Average: 0.615"
print "June Average: 0.501"
print "July Average year 1: 0.355"
print "July Average year 2: 0.316"
print "July Average year 3: 0.346"


print "Peak time average(6/20 - 8/31) year 1: 0.365"
print "Peak time Average(7/1 - 8/31) year 3: 0.304"
print "Peak time Average(7/1 - 8/31) year 2: 0.25"



# # Solar, Wind, Battery Model

# In[18]:

data_solar = pd.read_csv("2010_solar.csv")

# Success - Display the first record
display(data_solar.head(n=10)) 



# In[19]:

score = []
usage = []
for j in range(1,8760):
    usage.append(data['Area1 Amagansett'][j] + data['Area2-Bridgehampton&W/OHither Hills'][j] + data['Area3-Canal&W/OE.Hampton/Buell'][j])
fig, ax1 = plt.subplots()

#plot total consumption of 3 neighborhoods

plt.plot_date(date[1:8760],usage,markersize=.1,linewidth=1, linestyle="-")
plt.gcf().autofmt_xdate()
ax1.set_xlim([datetime.date(2013, 1, 1), datetime.date(2013, 12, 30)])
ax1.set_ylim([0, 220])
plt.rcParams["figure.figsize"] = [20,9]

# set capacity of each source
wind = 150   #MW
solar = 40000 #m2
battery = 150 #MW

solar1 = []   #lists to store daily generation form each source
wind1 = []
battery1 = []
total_available = []


solar_eff = 0.15  # solar efficiency and performance 
solar_perform = 0.8

battery_storage = 0

for i in range(1,8760):
    solar_gen = solar * solar_eff * solar_perform * data_solar['METSTAT Glo (Wh/m^2)'][i-1] * 0.000001
    wind_gen = data_wind['off.K1'][i] * wind
    total_available.append(solar_gen + wind_gen + battery_storage)
    
    solar1.append(solar_gen)
    wind1.append(wind_gen)
    battery1.append(battery_storage)
    
    if usage[i-1] >= (wind_gen + solar_gen + battery_storage):
        battery_storage = 0
        score.append(0)
    
    elif usage[i-1] >= (wind_gen + solar_gen):
        battery_storage = battery_storage - (usage[i-1] - (wind_gen + solar_gen))
        score.append(1)

    elif usage[i-1] < (wind_gen + solar_gen):
        battery_storage = battery_storage + ((wind_gen + solar_gen) - usage[i-1])
        score.append(1)
        if battery_storage > battery:
            battery_storage = battery
            
            
ax2 = ax1.twinx()
#ax2.plot_date(date[1:8760],total_available,markersize=.1,linewidth=1, linestyle="-", color = 'r')
#ax2.plot_date(date[1:8760],solar1,markersize=.1,linewidth=1, linestyle="-", color = 'y')
ax2.plot_date(date[1:8760],wind1,markersize=.1,linewidth=1, linestyle="-", color = 'k')


ax2.set_xlim([datetime.date(2013, 1, 1), datetime.date(2013, 5, 15)])

ax2.set_ylim([-0.2, 300])


plt.rcParams["figure.figsize"] = [20,9]
plt.show()

length = len(score)
print "Total number of records: {}".format(length)

sc = np.mean(score)*100
print "Score: {}%".format(sc)




# # Wind Export - Import Calculation

# In[1]:

# imp, exp = imported and exported electricity in MW 

 #list to store hourly generated wind data
def only_wind(wind_cap, bat_cap, imp, exp, per, start, end): # per = how do u want to see the imp/exp data. hourly(8760)/daily(365)/yearly(1)? 
    wind = []
    usage_wind = []                                 # start - start year, end - end year, bat_cap = battery capacity
    x = 8760
    for j in range(0,8760*3):
        usage_wind.append(data['Area1 Amagansett'][j] + data['Area2-Bridgehampton&W/OHither Hills'][j] + data['Area3-Canal&W/OE.Hampton/Buell'][j])
   
    score = [] # to keep score of success
    battery_storage = [] # available storage
    battery_used = [] # usable storage
    battery = 0
    imprt = []
    exprt = []
    imp = 0
    exp = 0
    for i in range(start*x,end*x):
        wind_gen = data_wind['off.K1'][i] * wind_cap
        wind.append(wind_gen)
        temp1 = battery                     # to store available power BEFORE each hour
        
        if usage_wind[i] >= (wind_gen + battery):
            battery = battery - bat_cap*(7/13.5)
            if battery < 0:
                battery = 0
            temp2 = battery                 # to store available power AFTER each hour
            imp = usage_wind[i] - (wind_gen + (temp1-temp2))
            imprt.append(imp)
            exprt.append(0)
            battery_used.append(temp1-temp2)
            score.append(0)
        
        elif usage_wind[i] >= (wind_gen):
            reqd = usage_wind[i] - wind_gen
            if reqd > bat_cap*(7/13.5):
                battery = battery - bat_cap*(7/13.5)
                temp2 = battery
                imp = usage_wind[i] - (wind_gen + (temp1-temp2)) 
                imprt.append(imp)
                battery_used.append(temp1-temp2)
                score.append(0)
            else:
                battery = battery - reqd
                battery_used.append(reqd)
                score.append(1)
                exprt.append(0)
                imprt.append(0)
                
            

        elif usage_wind[i] < (wind_gen):
            avlbl = min(bat_cap - battery, bat_cap*(5/13.5))  # avlbl = available storage
            battery_used.append(0)
            if wind_gen - usage_wind[i] > avlbl:
                battery = battery + avlbl
                temp2 = battery
                exp = (wind_gen - usage_wind[i] - (temp2-temp1))
                exprt.append(exp)
                imprt.append(0)
            else:
                battery = battery + wind_gen - usage_wind[i]
                imprt.append(0)
                exprt.append(0)
            score.append(1)
        
        battery_storage.append(battery)

        
    imp_monthly = [] #list for monthly import and export
    exp_monthly = []       
    usage_monthly = []
    wind_monthly = []
    battery_used_monthly = []
    battery_storage_monthly = []
    for k in range(1 + 12*start, 1+ 12*end):
        if k == 1:
            st = 0
            n_days = 31
        if k == 2:
            st = 30
            n_days = 28
        if k == 3:
            st = 58
            n_days = 31
        if k == 4:
            st = 89
            n_days = 30
        if k == 5:
            st = 119
            n_days = 31
        if k == 6:
            st = 150
            n_days = 30
        if k == 7:
            st = 180
            n_days = 31
        if k == 8:
            st = 211
            n_days = 31
        if k == 9:
            st = 242
            n_days = 30
        if k == 10:
            st = 272
            n_days = 31
        if k == 11:
            st = 303
            n_days = 30
        if k == 12:
            st = 333
            n_days = 31
        
        
        if k == 13:
            st = 364 + 0
            n_days = 31
        if k == 14:
            st = 364 + 30
            n_days = 28
        if k == 15:
            st = 364 + 58
            n_days = 31
        if k == 16:
            st = 364 + 89
            n_days = 30
        if k == 17:
            st = 364 + 119
            n_days = 31
        if k == 18:
            st = 364 + 150
            n_days = 30
        if k == 19:
            st = 364 + 180
            n_days = 31
        if k == 20:
            st = 364 + 211
            n_days = 31
        if k == 21:
            st = 364 + 242
            n_days = 30
        if k == 22:
            st = 364 + 272
            n_days = 31
        if k == 23:
            st = 364 + 303
            n_days = 30
        if k == 24:
            st = 364 + 333
            n_days = 31
    
        
        if k == 25:
            st = 728 + 0
            n_days = 31
        if k == 26:
            st = 728 + 30
            n_days = 28
        if k == 27:
            st = 728 + 58
            n_days = 31
        if k == 28:
            st = 728 + 89
            n_days = 30
        if k == 29:
            st = 728 + 119
            n_days = 31
        if k == 30:
            st = 728 + 150
            n_days = 30
        if k == 31:
            st = 728 + 180
            n_days = 31
        if k == 32:
            st = 728 + 211
            n_days = 31
        if k == 33:
            st = 728 + 242
            n_days = 30
        if k == 34:
            st = 728 + 272
            n_days = 31
        if k == 35:
            st = 728 + 303
            n_days = 30
        if k == 36:
            st = 728 + 333
            n_days = 31
        
        temp_imp = 0 #storing import and export values
        temp_exp = 0
        temp_battery_storage = 0
        temp_battery_used = 0
        use = 0 #storing monthly usage values
        wind_mon = 0
        
        
        for i in range(st, st + n_days):
            t = i*24
            for j in range(t,t+24):
                temp_imp = temp_imp + imprt[j-start*x]
                temp_exp = temp_exp + exprt[j-start*x]
                temp_battery_used = temp_battery_used + battery_used[j-start*x]
                temp_battery_storage = temp_battery_storage + battery_storage[j-start*x]
                use = use + usage_wind[j]
                wind_mon = wind_mon + wind[j-start*x]
                
                
        wind_monthly.append(wind_mon)
        usage_monthly.append(use)
        imp_monthly.append(temp_imp)
        exp_monthly.append(temp_exp)
        battery_used_monthly.append(temp_battery_used)
        battery_storage_monthly.append(temp_battery_storage/n_days)

    x = []
    
    
    for year in range(2013 + start, 2013 + end):
        for month in range(1, 13):
            x.append(dt.datetime(year=year, month=month, day=1))
        
    plt.plot_date(x,exp_monthly,color='b',markersize=.1,linewidth=1.5, linestyle="-")
    plt.plot_date(x,imp_monthly,color='r',markersize=.1,linewidth=1.5, linestyle="-")
    plt.plot_date(x,usage_monthly,color='k',markersize=.1,linewidth=1.5, linestyle="-")
    plt.plot_date(x,wind_monthly,color='g',markersize=.1,linewidth=1.5, linestyle="-")
    plt.plot_date(x,battery_used_monthly,color='c',markersize=.1,linewidth=1.5, linestyle="-")
    plt.plot_date(x,battery_storage_monthly,color='m',markersize=.1,linewidth=1.5, linestyle="-")

    plt.show()
    sc = np.mean(score)*100
    print "Score: {}%".format(sc)
  
    print imp_monthly
    print exp_monthly
    return sum(wind), sum(usage_monthly), sum(imp_monthly), sum(exp_monthly)



ans =  only_wind(165.2, 0, 0, 0, 365, 0, 1)
print "Wind: {} MWh".format(ans[0])
print "Usage: {} MWh".format(ans[1])
print "Import: {} MWh".format(ans[2])
print "Export: {} MWh".format(ans[3])
ans =  only_wind(165.2, 400, 0, 0, 365, 0, 1)
print "Wind: {} MWh".format(ans[0])
print "Usage: {} MWh".format(ans[1])
print "Import: {} MWh".format(ans[2])
print "Export: {} MWh".format(ans[3])
ans =  only_wind(165.2, 900, 0, 0, 365, 0, 1)
print "Wind: {} MWh".format(ans[0])
print "Usage: {} MWh".format(ans[1])
print "Import: {} MWh".format(ans[2])
print "Export: {} MWh".format(ans[3])
ans =  only_wind(165.2, 1500, 0, 0, 365, 0, 1)
print "Wind: {} MWh".format(ans[0])
print "Usage: {} MWh".format(ans[1])
print "Import: {} MWh".format(ans[2])
print "Export: {} MWh".format(ans[3])


# #Optimal Values

# 0-3 average = 186.04
# Import: 920891.586408 MWh
# Export: 920863.089873 MWh
# 
# 0-1 average = 165.2
# Import: 288323.412437 MWh
# Export: 288244.155235 MWh
# 
# 1-2 average = 187.45
# Import: 303589.732762 MWh
# Export: 303567.508139 MWh
# 
# 2-3 average = 209.4
# Import: 326093.443019 MWh
# Export: 326107.539335 MWh
# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



