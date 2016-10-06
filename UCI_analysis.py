import pandas as pd 
import numpy as np
import pdb
import matplotlib.pyplot as pl
import scipy.stats as sst


def HypTest(data_m,aud_no=None):

	##  Hiermit muss wirklich sehr vorsichtig umgegangen werden. Die Statistiken sind eben NICHT normalverteilt.
	##  Generell gilt GROSSES P == GROSSE UEBEREINSTIMMUNG 	
	print('\n    Sample Size: %i' %len(data_m))
	if aud_no == None:
		temp = data_m[['av_day_aud_reg','av_day_aud_dyn']]
	else:
		temp = data_m[['av_day_aud_reg','av_day_aud_dyn']][data_m.auditorium_no_x == aud_no]
	temp.rename(columns= {'av_day_aud_reg':'av_reg','av_day_aud_dyn':'av_dyn'},inplace=True)

	# Calculate parameters
	reg_td_av, dyn_td_av = temp.mean()[:2]
	print('\n\n WE WANT LARGE P!! \n')
	print('\n    MeanValues d/r %5.4f   %5.4f \n    Difference Mean Values:  %5.4f' %(dyn_td_av, reg_td_av,reg_td_av-dyn_td_av))
	n = len(temp.av_reg)
	reg_dev = np.sqrt(((temp.av_reg - reg_td_av)**2).values.sum()/(len(temp.av_reg)-1))
	dyn_dev = np.sqrt(((temp.av_dyn - dyn_td_av)**2).values.sum()/(len(temp.av_dyn)-1))
	print('\n    Standart Deviation Regular:  %5.4f' %(reg_dev))
	print('\n    Standart Deviation Dynamic:  %5.4f' %(dyn_dev))

	td,pd = sst.kstest(temp.av_dyn, 'norm',args=(dyn_td_av,dyn_dev))
	tr,pr = sst.kstest(temp.av_reg, 'norm',args=(reg_td_av,reg_dev))
	print('\n    output normaltest for regular:  t = %g  p = %g' % (tr,pr)) 
	print('\n    output normaltest for dynamic:  t = %g  p = %g' % (td,pd)) 
	
	tc,pc = sst.kstest(temp.av_dyn, 'norm',args=(reg_td_av,reg_dev),alternative='less')
	print('\n    COMPARE both distributions IF BOTH ARE NORMAL DISTRIBUTIONS:  t = %g  p = %g' % (tc,pc))

	return temp
	

def plot_weekly_total(weekly_total):
	
	## Plottet die woechentliche Entwicklung von regular und dynamic pricing fuer das komplette Kino	

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
	pl.text(25,11,'Mean reg/dyn: %6.5f / %6.5f' %(np.mean(av_reg),np.mean(av_dyn)),fontsize=15)
	pl.show()
	

def plot_weekly_aud(weekly_aud, aud_no):

	## Plottet die woechentliche Entwicklung von regular und dynamic pricing fuer die EINZELNE AUDITORIUMS
	
	aud_cap = [None,187,299,431,167,184,326,298,187]
	temp =  weekly_aud[['av_reg','av_dyn','tickets_per_day_per_aud','tickets_per_day_per_aud_dynamic']][weekly_aud.auditorium_no == aud_no]
	temp['diff'] = temp.av_dyn - temp.av_reg
	x = temp.index
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

class analyze_UCI:

	## data input muss 'show_start','auditorium_no', 'seat_row','ext_show_id' enthalten
 
	def __init__(self,data):
		#data = pd.read_csv('2015_2016_clean_uci_potsdam.csv', usecols=['show_start','auditorium_no','seat_number', 'seat_row'],parse_dates=['show_start'],nrows=1000)
		self.data = data
		self.calculate_capacities() 

		self.regular = self.data[self.data.show_start<pd.datetime(2015,8,12).date()]
		self.dynamic = self.data[(self.data.show_start>=pd.datetime(2016,3,31).date()) & (self.data.show_start<pd.datetime(2016,8,12).date())]
		self.regular = self.regular.merge(self.shows_aus[['Einteilung','ext_show_id']])
		self.dynamic = self.dynamic.merge(self.shows_aus[['Einteilung','ext_show_id']])
		self.selected_demand = 'no'

	def calculate_capacities(self,Einteilung={'low':0.150,'med':0.30}):
		
		## Berechnung der Auslastung der einzelnen Kinosaele 
		
		aud_cap = pd.DataFrame(data={'capacities':[187,299,431,167,184,326,298,187],'auditorium_no':[1,2,3,4,5,6,7,8]})
		shows_aus = pd.DataFrame(self.data.groupby(['ext_show_id','auditorium_no']).size())
		shows_aus.rename(columns= {0:'visitors'},inplace=True)
		shows_aus = shows_aus.reset_index()
		shows_aus = shows_aus.merge(aud_cap)
		shows_aus['Auslastung'] = shows_aus.visitors/shows_aus.capacities
		shows_aus['Einteilung'] = shows_aus.ext_show_id
		shows_aus['Einteilung'][shows_aus.Auslastung < Einteilung['low']] = 'low'
		shows_aus['Einteilung'][(shows_aus.Auslastung >= Einteilung['low']) & (shows_aus.Auslastung <= Einteilung['med'])] = 'med'
		shows_aus['Einteilung'][shows_aus.Auslastung > Einteilung['med']] = 'high'
		self.shows_aus =  shows_aus[['ext_show_id','Einteilung','Auslastung']]

	def demand(self,demand):

		## Ggf Unterscheidung zwischen HIGH/MED/LOW 		

		print('\n    Extract only %s demand shows' %demand)
		self.regular = self.regular[self.regular.Einteilung==demand]
		self.dynamic = self.dynamic[self.dynamic.Einteilung==demand]
		self.selected_demand = demand
	
	def un_demand(self):

		## Undo Demand Option

		self.regular = self.data[self.data.show_start<pd.datetime(2015,8,12).date()e
		self.dynamic = self.data[self.data.show_start>=pd.datetime(2016,3,31).date()]
		self.regular = self.regular.merge(self.shows_aus[['Einteilung','ext_show_id']])
		self.dynamic = self.dynamic.merge(self.shows_aus[['Einteilung','ext_show_id']])
		self.selected_demand = 'no'
		
	
	def calculate_average_rows_total(self):

		## Berechnung der Durschnittsreihe abhaenging Audtiorium

		self.avg_row_aud = pd.DataFrame(self.regular.groupby(['show_start','auditorium_no']).sum()['seat_row'])
		self.avg_row_aud_dyn = pd.DataFrame(self.dynamic.groupby(['show_start','auditorium_no']).sum()['seat_row'])
		self.avg_row_aud.rename(columns= {'seat_row':'tot_row_val'},inplace=True)
		self.avg_row_aud_dyn.rename(columns= {'seat_row':'tot_row_val_dynamic'},inplace=True)
		self.avg_row_aud['tickets_per_day_per_aud'] = pd.DataFrame(self.regular.groupby(['show_start','auditorium_no']).size())
		self.avg_row_aud_dyn['tickets_per_day_per_aud_dynamic'] = pd.DataFrame(self.dynamic.groupby(['show_start','auditorium_no']).size())

	def merge_week(self):
	
		## Zusammenfassung der Ergebnisse auf Wochenbasis
		
		weekly_aud_reg = self.avg_row_aud.reset_index()
		weekly_aud_reg = weekly_aud_reg.groupby([pd.to_datetime(weekly_aud_reg.show_start).dt.week,'auditorium_no']).sum()
		weekly_aud_dyn = self.avg_row_aud_dyn.reset_index()
		weekly_aud_dyn = weekly_aud_dyn.groupby([pd.to_datetime(weekly_aud_dyn.show_start).dt.week,'auditorium_no']).sum()

		weekly_aud_reg = weekly_aud_reg.reset_index()
		weekly_aud_dyn = weekly_aud_dyn.reset_index()
		weekly_aud = weekly_aud_reg.merge(weekly_aud_dyn)
		weekly_total = weekly_aud.groupby('show_start').sum()

		weekly_aud['av_reg'] = weekly_aud.tot_row_val/ weekly_aud.tickets_per_day_per_aud
		weekly_aud['av_dyn'] = weekly_aud.tot_row_val_dynamic/ weekly_aud.tickets_per_day_per_aud_dynamic

		weekly_total['av_reg'] = weekly_total.tot_row_val/ weekly_total.tickets_per_day_per_aud
		weekly_total['av_dyn'] = weekly_total.tot_row_val_dynamic/ weekly_total.tickets_per_day_per_aud_dynamic
		weekly_total.drop(['auditorium_no'],axis=1,inplace=True)
		weekly_total.rename(columns= {'tickets_per_day_per_aud':'tickets_per_week','tickets_per_day_per_aud_dynamic':'tickets_per_week_dynamic'},inplace=True)
		
		self.weekly_total = weekly_total
		self.weekly_aud = weekly_aud

	def get_hypothese_data(self):

		## Berechnung der Daten auf Tagesebene um Input fuer Hypothesen Funktion (siehe oben) zu generieren

		data_r = pd.DataFrame(self.avg_row_aud.tot_row_val/ self.avg_row_aud.tickets_per_day_per_aud)
		data_r.rename(columns= {0:'av_day_aud_reg'},inplace=True)
		data_r = data_r.reset_index()
		data_r.show_start = pd.to_datetime(data_r.show_start)
		data_r['ind'] = 1000*data_r.show_start.dt.month+10*data_r.show_start.dt.day+data_r.auditorium_no

		data_d = pd.DataFrame(self.avg_row_aud_dyn.tot_row_val_dynamic/ self.avg_row_aud_dyn.tickets_per_day_per_aud_dynamic)
		data_d.rename(columns= {0:'av_day_aud_dyn'},inplace=True)
		data_d = data_d.reset_index()
		data_d.show_start = pd.to_datetime(data_d.show_start)
		data_d['ind'] = 1000*data_d.show_start.dt.month+10*data_d.show_start.dt.day+data_d.auditorium_no

		self.data_m = data_r.merge(data_d, on='ind')  
		self.data_m.drop(['auditorium_no_y','show_start_x','ind'],axis=1,inplace=True)

if __name__ == '__main__':
	data = pd.read_csv('2015_2016_clean_uci_potsdam.csv', usecols=['show_start','auditorium_no', 'seat_row','ext_show_id'],parse_dates=['show_start'])
	data = data.sort_values(by='show_start')
	data.show_start = data.show_start.dt.date
	
	UCI_a = analyze_UC(data)
	UCI_a.demand('low')
	UCI_a.calculate_average_rows_total()
	UCI_a.merge_week()
	UCI_a.get_hypothese_data()
	HypTest(UCI_a.data_m,aud_no=1)
	plot_weekly_aud(UCI_a.weekly_aud,3)
