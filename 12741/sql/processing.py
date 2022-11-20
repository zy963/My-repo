import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import TimeSeriesSplit
plt.style.use('ggplot')
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

accidents = pd.read_csv('CRASH_2020_Statewide.csv',usecols=['CRN','COLLISION_TYPE','CRASH_MONTH','CRASH_YEAR','DAY_OF_WEEK','DEC_LAT','DEC_LONG','FATAL_COUNT','HOUR_OF_DAY','ILLUMINATION','INJURY_COUNT','MAX_SEVERITY_LEVEL','TIME_OF_DAY','WEATHER1','WEATHER2','VEHICLE_COUNT','URBAN_RURAL'])
accidents.head()
accidents = accidents.dropna()
accidents.info()
accidents['TIME_OF_DAY'] = accidents['TIME_OF_DAY'].astype(str).str.zfill(4)
accidents['WEATHER1'] = accidents['WEATHER1'].astype(np.int64)
accidents['WEATHER2'] = accidents['WEATHER2'].astype(np.int64)
accidents.head()
# accidents = accidents.drop([accidents['WEATHER1']=='  '])
accidents.drop(accidents[accidents['WEATHER1']=='  '].index, inplace = True)
accidents.drop(accidents[accidents['WEATHER2']=='  '].index, inplace = True)
accidents_pitts = accidents[accidents['DEC_LAT'] >= 40.2]
accidents_pitts = accidents_pitts[accidents['DEC_LAT'] <= 40.7]
accidents_pitts = accidents_pitts[accidents['DEC_LONG'] <= -79.7]
accidents_pitts = accidents_pitts[accidents['DEC_LONG'] >= -80.4]

rainfall_station = pd.read_csv('3RWW_Rain_Gauges.csv',usecols = ['ID','NAME','ADDRESS','Y','X'])
rainfall_station = rainfall_station.rename(columns={'Y': 'LAT', 'X': 'LON'})
rainfall_station
# rainfall_station.to_csv('rainfall_station.csv')

pitts_2020 = pd.read_csv('2020_pitts.csv')
pitts_2019 = pd.read_csv('2019_pitts.csv')
pitts_all = pd.concat([pitts_2019, pitts_2020])
CRASH_VAR = ['CRASH_CRN','COLLISION_TYPE','CRASH_MONTH','CRASH_YEAR','DAY_OF_WEEK','DEC_LAT','DEC_LONG','FATAL_COUNT','HOUR_OF_DAY','ILLUMINATION','INJURY_COUNT','MAX_SEVERITY_LEVEL','TIME_OF_DAY','WEATHER','VEHICLE_COUNT','URBAN_RURAL']
FLAG_VAR = [
'WET_ROAD',               
'SNOW_SLUSH_ROAD',        
'ICY_ROAD',               
'ALCOHOL_RELATED', 
'DRINKING_DRIVER', 
'UNLICENSED',
'CELL_PHONE',
'SPEEDING_RELATED',
'AGGRESSIVE_DRIVING', 
'FATIGUE_ASLEEP']

ROADWAY_VAR = ['LANE_COUNT','SPEED_LIMIT']
roadways_2020 = pd.read_csv('ROADWAY_2020_Statewide.csv',usecols = ['CRN','ADJ_RDWY_SEQ','LANE_COUNT','SPEED_LIMIT'])
roadways_2019 = pd.read_csv('ROADWAY_2019_Statewide.csv',usecols = ['CRN','ADJ_RDWY_SEQ','LANE_COUNT','SPEED_LIMIT'])
roadways = pd.concat([roadways_2019,roadways_2020])

#read all csv rainfall file in pixel-level
import glob,os
from itertools import cycle

dic = ['1903','1904','1905','1906','1907','1908','1909','1910','1911','1912'
      '2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012']

pixel_cycle = cycle(pixel_list)
cols = ['PIXEL','month','amount']
# df = pd.DataFrame(index=range(len(dic) * len(pixel_list)),columns=cols)
df = pd.DataFrame(columns=cols)

for i in range(len(dic)):
    path=r'C:\Users\vic_h\OneDrive\桌面\Data Management Project\rainfall\rainfall\rainfall_' + dic[i]
    file_name=glob.glob(os.path.join(path, "R"+dic[i]+"_*.csv"))

#     print(file_name)
    rainfall_list = []
    for f in file_name:
        file = pd.read_csv(f)
        a = sum(file.iloc[:,5])
        rainfall_list.append(a)

    df_temp = pd.DataFrame(index=range(len(pixel_list)),columns=cols)
    df_temp['PIXEL'] = [next(pixel_cycle) for p in range(len(pixel_list))]
    df_temp['month'] = dic[i]
    df_temp['amount'] = rainfall_list

    df = pd.concat([df,df_temp])
    print("process " + str(i) + " month!")

    dic = ['1912',
      '2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012']

pixel_cycle = cycle(pixel_list)
cols = ['PIXEL','month','amount']
# df = pd.DataFrame(index=range(len(dic) * len(pixel_list)),columns=cols)
df2 = pd.DataFrame(columns=cols)

for i in range(len(dic)):
    path=r'C:\Users\vic_h\OneDrive\桌面\Data Management Project\rainfall\rainfall\rainfall_' + dic[i]
    file_name=glob.glob(os.path.join(path, "R"+dic[i]+"_*.csv"))

#     print(file_name)
    rainfall_list = []
    for f in file_name:
        file = pd.read_csv(f)
        a = sum(file.iloc[:,5])
        rainfall_list.append(a)

    df_temp = pd.DataFrame(index=range(len(pixel_list)),columns=cols)
    df_temp['PIXEL'] = [next(pixel_cycle) for p in range(len(pixel_list))]
    df_temp['month'] = dic[i]
    df_temp['amount'] = rainfall_list

    df2 = pd.concat([df2,df_temp])
    print("process " + str(i) + " month!")

pitts_all_final_delete = pitts_all_final.loc[~(((pitts_all_final['CRASH_YEAR'] == 2019) & (pitts_all_final['CRASH_MONTH'] == 1))
                                            | ((pitts_all_final['CRASH_YEAR'] == 2019) & (pitts_all_final['CRASH_MONTH'] == 2)))]

accident_new['YEAR'] = accident_new['CRASH_YEAR'].astype(str)
accident_new['YEAR'] = accident_new['YEAR'].str[2:]
accident_new['MONTH'] =  accident_new['CRASH_MONTH'].astype(str)
accident_new['MONTH'] = accident_new['MONTH'].apply(lambda x: x.zfill(2))
accident_new['TIME'] = accident_new['YEAR'] + accident_new['MONTH']


#multinomial logistic regression

#Computing the correlation matrix
corr = accident_all_together_use_attribute.corr()

#Generating a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Setting up the matplotlib figure
f, ax = plt.subplots(figsize=(8,8))

#Generating a custom diverging colormap
cmap = sns.diverging_palette(220,10, as_cmap=True)

#Drawing the heatmap with the mask
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.title("Correlation Matrix")
plt.show()

#plot distribution of variable
accident_all_together_plot = accident_all_together_use_attribute[['COLLISION_TYPE','DAY_OF_WEEK','HOUR_OF_DAY','ILLUMINATION'
                                                                ,'INJURY_COUNT','MAX_SEVERITY_LEVEL','TIME_OF_DAY','WEATHER'
                                                                ,'SPEED_LIMIT','amount']]
accident_all_together_plot.hist(bins = 50, figsize = (20,15))
plt.show()

#drop unknown variable
accident_final = accident_all_together[accident_all_together['COLLISION_TYPE'] != 98]
accident_final = accident_final[accident_final['COLLISION_TYPE'] != 99]
accident_final = accident_final[accident_final['HOUR_OF_DAY'] != 98]
accident_final = accident_final[accident_final['HOUR_OF_DAY'] != 99]
accident_final = accident_final[accident_final['WEATHER'] != 98]
accident_final = accident_final[accident_final['WEATHER'] != 99]
accident_final_severity = accident_final[accident_final['MAX_SEVERITY_LEVEL'] != 8]
accident_final_severity = accident_final_severity[accident_final_severity['MAX_SEVERITY_LEVEL'] != 9]

plt.figure(figsize = (10,12))
sns.scatterplot(data=accident_final_severity, x = "DEC_LONG", y = "DEC_LAT", hue = "MAX_SEVERITY_LEVEL", palette = "flare")
plt.show()

accident_final_severity.plot(kind = "scatter", x = "DEC_LONG", y = "DEC_LAT", alpha = 0.5,
             s = accident_final_severity["INJURY_COUNT"]*10, label = "INJURY_COUNT", figsize=(15,15),
             c = "MAX_SEVERITY_LEVEL", cmap = plt.get_cmap("YlGnBu"), colorbar= True
             )
plt.legend()

#Prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import log_loss

accident_ml = accident_final_severity.drop(['CRASH_CRN','MAX_SEVERITY_LEVEL','DEC_LAT','DEC_LONG','FATAL_COUNT','DRINKING_DRIVER','TIME_OF_DAY','TIME','PIXEL','month'],axis=1,inplace = False)
# accident_ml = accident_final_severity[['COLLISION_TYPE' , 'DAY_OF_WEEK' ,'ILLUMINATION', 'INJURY_COUNT','TIME_OF_DAY',
#                                        'WEATHER' , 'VEHICLE_COUNT' , 'Road_Surface_Conditions'
#                           , 'Light_Conditions', 'Sex_of_Driver' ,'Speed_limit']]

# Split the data into a training and test set.
X_train, X_test, y_train, y_test = train_test_split(accident_ml.values, 
                                              accident_final_severity['MAX_SEVERITY_LEVEL'].values,test_size=0.20, random_state=99)

for p in [0.005,0.01,0.05, 0.1, 1.0]:
    lr = LogisticRegression(penalty='l2', C=p,class_weight="balanced")
    #,class_weight="auto"
    # Fit the model on the trainng data.
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    sk_report = classification_report(
        digits=6,
        y_true=y_test, 
        y_pred=y_pred)
    print("Accuracy", round(accuracy_score(y_pred, y_test)*100,2))
    print(sk_report)
    pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)