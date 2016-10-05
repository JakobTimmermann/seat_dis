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

avg_row = pd.DataFrame(regular.groupby(['show_start']).sum()['seat_row'])
avg_row_dyn = pd.DataFrame(dynamic.groupby(['show_start']).sum()['seat_row'][:-1])

avg_row.rename(columns= {'seat_row':'avg_row_per_day'},inplace=True)
avg_row_dyn.rename(columns= {'seat_row':'avg_row_per_day'},inplace=True)
avg_row['tickets_per_day'] = pd.DataFrame(regular.groupby(['show_start']).size())
avg_row_dyn['tickets_per_day'] = pd.DataFrame(dynamic.groupby(['show_start']).size())

#avg_row['dyn']=avg_row_dyn.dyn.values
#avg_row['difference'] = avg_row.dyn.values-avg_row.reg.values

weekly = avg_row.reset_index()
weekly = weekly.groupby(pd.to_datetime(weekly.show_start).dt.week).sum()
weekly_dyn = avg_row_dyn.reset_index()
weekly_dyn = weekly_dyn.groupby(pd.to_datetime(weekly_dyn.show_start).dt.week).sum()
weekly = weekly.reset_index()
weekly_dyn = weekly_dyn.reset_index()
weekly['av_reg'] = weekly.avg_row_per_day/ weekly.tickets_per_day
weekly['av_dyn'] = weekly_dyn.avg_row_per_day/ weekly_dyn.tickets_per_day
weekly.plot()

## Different Auditoriums
avg_row_aud = pd.DataFrame(regular.groupby(['show_start','auditorium_no']).sum()['seat_row'])
avg_row_aud_dyn = pd.DataFrame(dynamic.groupby(['show_start','auditorium_no']).sum()['seat_row'][:-1])
avg_row_aud.rename(columns= {'seat_row':'avg_row_per_day'},inplace=True)
avg_row_aud_dyn.rename(columns= {'seat_row':'avg_row_per_day'},inplace=True)
avg_row_aud['tickets_per_day'] = pd.DataFrame(regular.groupby(['show_start','auditorium_no']).size())
avg_row_aud_dyn['tickets_per_day'] = pd.DataFrame(dynamic.groupby(['show_start','auditorium_no']).size())

weekly_aud = avg_row_aud.reset_index()
weekly_aud = weekly_aud.groupby([pd.to_datetime(weekly_aud.show_start).dt.week,'auditorium_no']).sum()
weekly_dyn_aud = avg_row_aud_dyn.reset_index()
weekly_dyn_aud = weekly_dyn_aud.groupby([pd.to_datetime(weekly_dyn_aud.show_start).dt.week,'auditorium_no']).sum()
weekly_aud = weekly_aud.reset_index()
weekly_dyn_aud = weekly_dyn_aud.reset_index()
weekly_aud['av_reg'] = weekly_aud.avg_row_per_day/ weekly_aud.tickets_per_day
weekly_aud['av_dyn'] = weekly_dyn_aud.avg_row_per_day/ weekly_dyn_aud.tickets_per_day



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

