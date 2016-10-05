import pandas as pd 
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats as sst


def HypTest(data):
	# Testing for normal distribution
	print('    output normaltest for regular') 
	print(sst.normaltest(data.av_reg))
	print('    output normaltest for dynamic') 
	print(sst.normaltest(data.av_dyn))

	# Zweistichproben Gauss Test : https://de.wikipedia.org/wiki/Gau%C3%9F-Test
	# 
	reg_td_av, dyn_td_av = data.mean()[:2]
	n = len(data.av_reg)
	reg_dev = np.sqrt(((data.av_reg - reg_td_av)**2).values.sum()/(len(data.av_reg)-1))
	dyn_dev = np.sqrt(((data.av_dyn - dyn_td_av)**2).values.sum()/(len(data.av_dyn)-1))

	Z=(dyn_td_av-reg_td_av)/np.sqrt((dyn_dev)**2/n + (reg_dev)**2/n)

	Z_dyn = np.sqrt(n)*(dyn_td_av - reg_td_av)/dyn_dev

	print('\n\n    Zweistichproben Gauss Test Z Value: %5.3f' %Z_dyn)

	# Welch-Test https://en.wikipedia.org/wiki/Welch%27s_t-test
	#
	#sst.ttest_ind_from_stats(reg_td_av,reg_dev,134,dyn_td_av,dyn_dev,134)


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


## Different Auditoriums
avg_row_aud = pd.DataFrame(regular.groupby(['show_start','auditorium_no']).sum()['seat_row'])
avg_row_aud_dyn = pd.DataFrame(dynamic.groupby(['show_start','auditorium_no']).sum()['seat_row'][:-1])
avg_row_aud.rename(columns= {'seat_row':'tot_row_val'},inplace=True)
avg_row_aud_dyn.rename(columns= {'seat_row':'tot_row_val_dynamic'},inplace=True)
avg_row_aud['tickets_per_day_per_aud'] = pd.DataFrame(regular.groupby(['show_start','auditorium_no']).size())
avg_row_aud_dyn['tickets_per_day_per_aud_dynamic'] = pd.DataFrame(dynamic.groupby(['show_start','auditorium_no']).size())

weekly_aud_reg = avg_row_aud.reset_index()
weekly_aud_reg = weekly_aud_reg.groupby([pd.to_datetime(weekly_aud_reg.show_start).dt.week,'auditorium_no']).sum()
weekly_dyn_aud = avg_row_aud_dyn.reset_index()
weekly_dyn_aud = weekly_dyn_aud.groupby([pd.to_datetime(weekly_dyn_aud.show_start).dt.week,'auditorium_no']).sum()

weekly_aud_reg = weekly_aud_reg.reset_index()
weekly_dyn_aud = weekly_dyn_aud.reset_index()
weekly_aud = weekly_aud_reg.merge(weekly_dyn_aud)
weekly_total = weekly_aud.groupby('show_start').sum()

weekly_aud['av_reg'] = weekly_aud.tot_row_val/ weekly_aud.tickets_per_day_per_aud
weekly_aud['av_dyn'] = weekly_aud.tot_row_val_dynamic/ weekly_aud.tickets_per_day_per_aud_dynamic

weekly_total['av_reg'] = weekly_total.tot_row_val/ weekly_total.tickets_per_day_per_aud
weekly_total['av_dyn'] = weekly_total.tot_row_val_dynamic/ weekly_total.tickets_per_day_per_aud_dynamic

aud_no = 3
temp =  weekly_aud[['av_reg','av_dyn']][weekly_aud.auditorium_no == aud_no]
temp['diff'] = temp.av_dyn - temp.av_reg
temp.plot()

data = pd.DataFrame(avg_row_aud.tot_row_val/ avg_row_aud.tickets_per_day_per_aud)
data.rename(columns= {0:'av_reg_day_aud'},inplace=True)
data = data.reset_index()
data.show_start = pd.to_datetime(data.show_start)
data['ind'] = 1000*data.show_start.dt.month+10*data.show_start.dt.day+data.auditorium_no

data_d = pd.DataFrame(avg_row_aud_dyn.tot_row_val_dynamic/ avg_row_aud_dyn.tickets_per_day_per_aud_dynamic)
data_d.rename(columns= {0:'av_reg_day_aud_dyn'},inplace=True)
data_d = data_d.reset_index()
data_d.show_start = pd.to_datetime(data_d.show_start)
data_d['ind'] = 1000*data_d.show_start.dt.month+10*data_d.show_start.dt.day+data_d.auditorium_no

data = data.merge(data_d, on='ind')  
data.drop(['auditorium_no_y','show_start_x','ind'],axis=1,inplace=True)

temp = data[['av_reg_day_aud','av_reg_day_aud_dyn']][data.auditorium_no_x == aud_no]
temp.rename(columns= {'av_reg_day_aud':'av_reg','av_reg_day_aud_dyn':'av_dyn'},inplace=True)

HypTest(temp)

plt.show()


'''
>>>>>>> master
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
'''
