import pandas as pd 
import numpy as np
import pdb
import matplotlib.pyplot as pl
import scipy.stats as sst


def HypTest(data,aud_no):
	temp = data_m[['av_reg_day_aud','av_reg_day_aud_dyn']][data_m.auditorium_no_x == aud_no]
	temp.rename(columns= {'av_reg_day_aud':'av_reg','av_reg_day_aud_dyn':'av_dyn'},inplace=True)
	# Testing for normal distribution
	print('    output normaltest for regular') 
	print(sst.normaltest(data.av_reg))
	print('    output normaltest for dynamic') 
	print(sst.normaltest(data.av_dyn))

	# Zweistichproben Gauss Test : https://de.wikipedia.org/wiki/Gau%C3%9F-Test
	# 
	reg_td_av, dyn_td_av = data.mean()[:2]
	print(reg_td_av)
	print(dyn_td_av)
	n = len(data.av_reg)
	reg_dev = np.sqrt(((data.av_reg - reg_td_av)**2).values.sum()/(len(data.av_reg)-1))
	dyn_dev = np.sqrt(((data.av_dyn - dyn_td_av)**2).values.sum()/(len(data.av_dyn)-1))
	print(reg_dev)
	print(dyn_dev)

	Z=(dyn_td_av-reg_td_av)/np.sqrt((dyn_dev)**2/n + (reg_dev)**2/n)

	Z_dyn = np.sqrt(n)*(dyn_td_av - reg_td_av)/dyn_dev

	print('\n\n    Zweistichproben Gauss Test Z Value: %5.3f' %Z_dyn)

	# Welch-Test https://en.wikipedia.org/wiki/Welch%27s_t-test
	#
	#sst.ttest_ind_from_stats(reg_td_av,reg_dev,134,dyn_td_av,dyn_dev,134)
	#sst.kstest()


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
weekly_total.drop(['auditorium_no'],axis=1,inplace=True)
weekly_total.rename(columns= {'tickets_per_day_per_aud':'tickets_per_week','tickets_per_day_per_aud_dynamic':'tickets_per_week_dynamic'},inplace=True)

def plot_weekly_total(weekly_total):
	x = weekly_total.index
	av_dyn, av_reg = weekly_total[['av_dyn','av_reg']].values.transpose()
	att_dyn, att_reg = weekly_total[['tickets_per_week_dynamic','tickets_per_week']].values.transpose()
	att_dyn, att_reg = att_dyn/max(att_reg)*4, att_reg/max(att_reg)*4
	pl.plot(x,av_dyn,label='average row dynamic pricing')
	pl.plot(x,av_reg,label='average row regular pricing')

	pl.plot(x,att_dyn,label='attendance dynamic')
	pl.plot(x,att_reg,label='attendance regular')
	pl.legend(loc='upper left')
	pl.ylim([0,12])
	pl.xlabel('Calender Week')
	pl.ylabel('row number')
	pl.show()
	

def plot_weekly_aud(weekly_aud, aud_no):
	aud_cap = [None,187,299,431,167,184,326,298,187]
	temp =  weekly_aud[['av_reg','av_dyn','tickets_per_day_per_aud','tickets_per_day_per_aud_dynamic']][weekly_aud.auditorium_no == aud_no]
	temp['diff'] = temp.av_dyn - temp.av_reg
	x = weekly_total.index
	av_dyn, av_reg = temp[['av_dyn','av_reg']].values.transpose()
	att_dyn, att_reg = temp[['tickets_per_day_per_aud_dynamic','tickets_per_day_per_aud']].values.transpose()
	att_dyn, att_reg = att_dyn/max(att_reg)*4, att_reg/max(att_reg)*4
	pl.plot(x,av_dyn,label='average row dynamic pricing')
	pl.plot(x,av_reg,label='average row regular pricing')

	pl.plot(x,att_dyn,label='attendance dynamic')
	pl.plot(x,att_reg,label='attendance regular')
	pl.legend(loc='upper left')
	pl.ylim([0,12])
	pl.xlabel('Calender Week')
	pl.ylabel('row number')
	pl.title('Auditorium %i'%aud_no, fontsize=18, fontweight='bold')
	pl.text(25,11,'Capacity: %i' %(aud_cap[aud_no]),fontsize=15)
	pl.show()

data_r = pd.DataFrame(avg_row_aud.tot_row_val/ avg_row_aud.tickets_per_day_per_aud)
data_r.rename(columns= {0:'av_reg_day_aud'},inplace=True)
data_r = data_r.reset_index()
data_r.show_start = pd.to_datetime(data_r.show_start)
data_r['ind'] = 1000*data_r.show_start.dt.month+10*data_r.show_start.dt.day+data_r.auditorium_no

data_d = pd.DataFrame(avg_row_aud_dyn.tot_row_val_dynamic/ avg_row_aud_dyn.tickets_per_day_per_aud_dynamic)
data_d.rename(columns= {0:'av_reg_day_aud_dyn'},inplace=True)
data_d = data_d.reset_index()
data_d.show_start = pd.to_datetime(data_d.show_start)
data_d['ind'] = 1000*data_d.show_start.dt.month+10*data_d.show_start.dt.day+data_d.auditorium_no

data_m = data_r.merge(data_d, on='ind')  
data_m.drop(['auditorium_no_y','show_start_x','ind'],axis=1,inplace=True)

#HypTest(data_m,1)

aud_cap = pd.DataFrame(data={'capacities':[187,299,431,167,184,326,298,187],'auditorium_no':[1,2,3,4,5,6,7,8]})
shows_aus = pd.DataFrame(data.groupby(['ext_show_id','auditorium_no']).size())
shows_aus.rename(columns= {0:'visitors'},inplace=True)
shows_aus = shows_aus.reset_index()
shows_aus = shows_aus.merge(aud_cap)
shows_aus['Auslastung'] = shows_aus.visitors/shows_aus.capacities


