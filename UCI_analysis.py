import pandas as pd 
import numpy as np
import pdb
import matplotlib.pyplot as plt

#data = pd.read_csv('2015_2016_clean_uci_potsdam.csv', usecols=['show_start','auditorium_no','seat_number', 'seat_row'],parse_dates=['show_start'],nrows=1000)
df = pd.read_csv('2015_2016_clean_uci_potsdam.csv',nrows=1) 

data = pd.read_csv('2015_2016_clean_uci_potsdam.csv', usecols=['show_start','auditorium_no', 'seat_row','seat_block','ext_show_id'],parse_dates=['show_start'])
data = data.sort_values(by='show_start')
data.show_start = data.show_start.dt.date
regular = data[data.show_start<pd.datetime(2015,8,12).date()]
dynamic = data[data.show_start>=pd.datetime(2016,3,31).date()]

avg_row = pd.DataFrame(regular.groupby(['show_start']).mean()['seat_row'])
avg_row_dyn = pd.DataFrame(dynamic.groupby(['show_start']).mean()['seat_row'][:-1])
avg_row.rename(columns= {'seat_row':'reg'},inplace=True)
avg_row_dyn.rename(columns= {'seat_row':'dyn'},inplace=True)
avg_row['dyn']=avg_row_dyn.dyn.values
avg_row['difference'] = avg_row.dyn.values-avg_row.reg.values

# Testing for normal distribution
print('    output normaltest for regular') 
print(sst.normaltest(avg_row.reg))
print('    output normaltest for dynamic') 
print(sst.normaltest(avg_row.dyn))

# Zweistichproben Gauss Test : https://de.wikipedia.org/wiki/Gau%C3%9F-Test
# 
reg_td_av, dyn_td_av = avg_row.mean()[:2]
n = len(avg_row.reg)
reg_dev = np.sqrt(((avg_row.reg - reg_td_av)**2).values.sum()/(len(avg_row.reg)-1))
dyn_dev = np.sqrt(((avg_row.dyn - dyn_td_av)**2).values.sum()/(len(avg_row.dyn)-1))

Z=(dyn_td_av-reg_td_av)/np.sqrt((dyn_dev)**2/n + (reg_dev)**2/n)

Z_dyn = np.sqrt(len(avg_row.dyn))*(dyn_td_av - reg_td_av)/dyn_dev

# Welch-Test https://en.wikipedia.org/wiki/Welch%27s_t-test
#
import scipy.stats as sst
sst.ttest_ind_from_stats(reg_td_av,reg_dev,134,dyn_td_av,dyn_dev,134)

weekly = avg_row.reset_index()
weekly = weekly.groupby(pd.to_datetime(weekly.show_start).dt.week).mean()
weekly.plot()

temp = data.groupby(['auditorium_no','ext_show_id']).size()
#cap = []
#for k in temp.index.levels[0]:
#    cap.append(temp[k].max())
aud_cap = pd.DataFrame(data={'capacities':[187,299,431,167,184,326,298,187]},index=temp.index.levels[0])
temp = temp.reset_index().merge(aud_cap.reset_index())
temp.rename(columns = {0:'Visitors'},inplace=True)
temp['Auslastung'] = temp.Visitors/temp.capacities

dynamic = dynamic.merge(temp[['Auslastung','ext_show_id']])
regular = regular.merge(temp[['Auslastung','ext_show_id']])

avg_row_tot = pd.DataFrame(data.groupby(['show_start']).mean()['seat_row'][:-1])
grain = avg_row_tot.reset_index()
grain.show_start = pd.to_datetime(grain.show_start).dt.date
grain = grain[(grain.show_start<pd.datetime(2016,3,1).date()) & (grain.show_start>pd.datetime(2015,5,1).date())]
grain = grain.groupby(pd.to_datetime(grain.show_start).dt.week).mean()
grain.plot()
average_row_aud = data.groupby(['show_start','auditorium_no']).mean()['seat_row'][:-1]

