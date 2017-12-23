
# coding: utf-8

# # Machine Learning on Parking Space Occupancy in Dresden
# 
# Can we predict the occupation of a parking space ([Centrum-Galerie](https://www.bing.com/maps/default.aspx?q=centrum+galerie+dresden&mkt=de&FORM=HDRSC4)) in Dresden?
# 
# Read about it here: http://mechlab-engineering.de/2015/03/vorhersage-der-parkhausbelegung-mit-offenen-daten/

# In[1]:


import pandas as pd
import numpy as np
import time

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

pd.options.mode.chained_assignment = None  # default='warn'


# Take the historical data from https://parkendd.de/dumps/

# ### Read the data and process it

# In[2]:


name = 'Centrum Galerie'


# In[3]:


data = pd.read_csv('dresdencentrumgalerie-2016.csv', names=['Datum','free'], index_col='Datum', parse_dates=True)
data.sort_index(inplace=True)
data.tail()


# In[4]:


data.plot()


# Calc the occupation ([ger] Belegung)

# In[5]:


data['Belegung'] = 100.0-(data.free/950.0*100.0)
data['Belegung'] = data['Belegung'].astype(int)
data.drop('free', axis=1, inplace=True)


# # Predict for the `parkingspot`

# Ok, take a look at the TimeSeries

# In[6]:


# function to plot timeseries with weekend
def plotbelegung(df, which, fromdate, todate):
    if fromdate=='':
        fromdate=df.index[0]
    if todate=='':
        todate=df.index[-1]

    weekend = df[fromdate:todate].index[df[fromdate:todate].index.weekday>4]
    ax = df[fromdate:todate].plot(figsize=(16,6), ylim=(0, 120),
                                  alpha=0.9, rot=0,
                                  title='Auslastung Parkplatz \'%s\' Dresden %s bis %s' % (which, fromdate, todate))
    
    if not df.index.freqstr: # Wenn DataFrame keine Frequenz hat
        ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(0,1,2,3,4,5,6),interval=1))
        ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(0),interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d.'))
        ax.xaxis.set_major_formatter(dates.DateFormatter('\nSa/So'))
        ax.xaxis.grid(True, which="minor")
        ax.xaxis.grid(False, which="major")
    
    plt.ylabel('%')
    
    for w in weekend[::10]:
        plt.axvline(w, c='k', alpha=0.08, zorder=-1)
        
    return plt


# In[7]:


plotbelegung(data['Belegung'], name, '2016-12', '2016-12')


# # Let's do some Machine Learning on that

# ## Features
# 
# * Uhrzeit
# * Wochentag
# * Monat
# * Verkaufsoffener Sonntag?
# * nächster Tag Feiertag? (Deutschland)
# * übernächster Tag auch noch Feiertag? (Deutschland)
# * Ferien (Deutschland)
# * spezielles Event in Galerien oder Stadt oder Kultur
# * nächster Tag Feiertag? (Tschechien)
# * übernächster Tag auch noch Feiertag? (Tschechien)
# * Ferien (Tschechien)
# * Wetter
# 
# Wir haben nicht alle Features. Aber wenigstens die wichtigsten.

# ### Wochentag
# 
# Es ist wichtig, ob Montag oder Samstag oder Sonntag ist.

# In[8]:


data['Wochentag'] = data.index.dayofweek


# ### Uhrzeit
# 
# Dann ist es natürlich extrem wichtig, ob es in der Nacht ist oder tagsüber.

# In[9]:


data['Uhrzeit'] = data.index.hour*60.0 + data.index.minute


# ### Verkaufsoffener Sonntage

# In[10]:


offeneSonntage = pd.to_datetime(['2016-12-04'])


# In[11]:


def isoffenersonntag(serie):
    isoffen = False
    for offene in offeneSonntage:
        if serie.date() == offene.date():
            isoffen=True

    if isoffen:
        return 1
    else:
        return 0


# In[12]:


sonntagsseries = pd.Series(data.index, name='offeneSonntage', index=data.index).apply(isoffenersonntag)
data['offenerSonntag'] = sonntagsseries


# In[13]:


data[data.offenerSonntag==1].head(5)


# ### Feiertage
# 
# Wenn ein Feiertag naht, dann gehen auch mehr Leute in die Stadt. Genauso nach einem Feiertag. Deshalb ist das ein Feature.
# 
# Get them from http://www.feiertage.net/frei-tage.php

# In[14]:


feiertage = pd.DataFrame()
for year in range(2016, 2019):
    feiertage = feiertage.append(pd.read_csv('Sachsen%i.csv' % year, index_col=0, parse_dates=True, sep=';', dayfirst=True))


# In[15]:


feiertage


# Mit [numpy.busday_count](http://docs.scipy.org/doc/numpy/reference/generated/numpy.busday_count.html) bekommen wir die Anzahl der Werktage bis zum nächsten Feiertag, weil die Leute ja vor einem langen Wochenende immer noch mal richtig einkaufen gehen.

# In[16]:


def shoppingdaystonextfeiertag(df):
    diffs = []
    for feiertag in feiertage.index:
        diff = np.busday_count(df.date(), feiertag.date(), weekmask='Mon Tue Wed Thu Fri Sat')
        #print('%s bis %s: %i Arbeitstage' % (feiertag, df, diff)) 
        diffs.append(diff)
        
    try:
        return min([d for d in diffs if d>=0])
    except:
        return 100 # wenn kein Feiertag gefunden


# In[17]:


feiertagseries = pd.Series(data.index, name='Feiertage', index=data.index).apply(shoppingdaystonextfeiertag)
data['bisFeiertag'] = feiertagseries


# In[18]:


def shoppingdaysafterfeiertag(df):
    diffs = []
    for feiertag in feiertage.index:
        diff = np.busday_count(feiertag.date(), df.date(), weekmask='Mon Tue Wed Thu Fri Sat')
        #print('%s bis %s: %i Arbeitstage' % (feiertag, df, diff)) 
        diffs.append(diff)
        
    try:
        return min([d for d in diffs if d>=0])
    except:
        return 100 # wenn kein Feiertag gefunden


# In[19]:


feiertagseries = pd.Series(data.index, name='Feiertage', index=data.index).apply(shoppingdaysafterfeiertag)
data['nachFeiertag'] = feiertagseries


# ### Schulferien
# 
# Schulferien Sachsen: http://www.schulferien.org/Sachsen/sachsen.html

# In[20]:


schulferien = [['2016-02-08','2016-02-19'],
                ['2016-03-25','2016-04-01'],
                ['2016-06-27','2016-08-05'],
                ['2016-10-03','2016-10-15'],
                ['2018-02-12','2018-02-23'],
                ['2018-05-19','2018-05-22'],
                ['2018-07-02','2018-08-10'],
                ['2018-10-08','2018-10-20'],
                ['2018-12-22','2019-01-04']]


# In[21]:


def isschulferien(series):
    now = series.date()
    
    inferien = False
    for ferien in schulferien:
        anfang = pd.to_datetime(ferien[0]).date()
        ende = pd.to_datetime(ferien[1]).date()

        if now >= anfang and now <= ende:
            inferien = True

    if inferien:
        return 1
    else:
        return 0


# In[22]:


ferienseries = pd.Series(data.index, name='Schulferien', index=data.index).apply(isschulferien)
data['Schulferien'] = ferienseries


# ### Weihnachten

# In[23]:


def isweihnachten(series):
    if series.month==12:
        return 1
    else:
        return 0


# In[24]:


weihnachtsseries = pd.Series(data.index, name='Weihnachten', index=data.index).apply(isweihnachten)
data['Weihnachten'] = weihnachtsseries


# ### Check

# In[25]:


data.groupby([data.index.year, data.index.month, data.index.day]).first()


# ### Featurevector

# In[26]:


featurevector = ['Wochentag','Uhrzeit','Schulferien','offenerSonntag','bisFeiertag','nachFeiertag','Weihnachten']


# OK, nun haben wir Labels und Features fertig.
# 
# Nun können wir Test und Trainingsdatensatz erstellen.
# 
# ## Train some Machine Learning Classifiers

# In[27]:


labels = data['Belegung'].values
np.shape(labels)


# In[28]:


features = data[featurevector].values
np.shape(features)


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


# In[30]:


from sklearn.tree import DecisionTreeRegressor


# ### Split into test and training set
# 
# To get an accuracy score, we need to split our dataset in a training and a test set. We train with the training set and test the model later with the part of the test set.

# In[31]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)


# In[32]:


np.shape(labels_test)


# ### DecisionTreeRegressor

# Because it is a regression (output of the prediction are integer) it might fit the data better than a classifier. So we test the the predicted labels with the test labels with the $R^2$ score (coefficient of determination).

# In[33]:


print('max_depth\tmin_samples_leaf\tR2_score (higher is better)')
maxscore = 0.8
for md in range(10, 30):
    for mins in range(20, 50):
        classifier = DecisionTreeRegressor(max_depth=md, min_samples_leaf=mins).fit(features_train, labels_train)
        labels_predict = classifier.predict(features_test)
        #labels_rounded = [round(label/10.0)*10.0 for label in labels_predict]
        score = r2_score(labels_test, labels_predict)
        if score>maxscore:
            print('%i\t\t\t%i\t\t\t%.3f' % (md, mins, score))
            maxscore=score


# A high $R^2$-score is better, but if we have a high `max_depth` and low `min_samples_leaf`, we might overfit the dataset. The score is not everything. See [Tips on Practical Use of Decision Trees](http://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use)

# ## Model

# In[34]:


classifier = DecisionTreeRegressor(max_depth=10, min_samples_leaf=20)
classifier.fit(features_train, labels_train)


# ### Overfit test
# 
# One can check for overfitting by just test the model with the training features. If the score is very high (and the score with the test features is low), it is likely, that the model is overfitted.

# In[35]:


# Let's check if we overfit. If, the accuracy with the training set is very high, with the test set very low.
labels_predict_overfittest = classifier.predict(features_train).astype('int')
score = accuracy_score(labels_train, labels_predict_overfittest)
print('%.3f Accuracy Score with training data' % (score))

if score>0.4:
    print('You probably overfit it!')


# ### What is the most important feature for the model

# In[36]:


importances = classifier.feature_importances_


# In[37]:


featureimportance = pd.DataFrame(index=featurevector, data=importances, columns=['Importance']).sort_values('Importance', ascending=False).plot(kind='bar', rot=20)


# Obviously it is the time of the day (`Minuten` since midnight), the calender week, the days until the next holiday (because people tend to go shopping if the weekend is long) and for sure the day of the week (because saturday is shopping day!).

# In[38]:


labels_predict = classifier.predict(features_test).astype('int')


# In[39]:


r2_score(labels_test, labels_predict)


# ### Let's take a look at the confusion matrix

# In[40]:


cm = confusion_matrix(labels_predict, labels_test)

plt.matshow(cm, cmap=plt.cm.Spectral_r, interpolation='none')
plt.colorbar(shrink=0.8)
plt.ylabel(u'Wahre Belegung in %')
plt.xlabel(u'Geschätzte Belegung in %')


# In[76]:


plt.scatter(labels_predict, labels_test, alpha=0.01)
plt.ylabel(u'Wahre Belegung in %')
plt.xlabel(u'Geschätzte Belegung in %')
plt.xlim([0, 110])
plt.ylim([0, 110])


# As you can see, the confusion matrix looks pretty good. That's not perfect, but take into account, that we just have a few features for such a complex scenario like inner-city parking space occupation.

# # Predict
# 
# Here we predict it for the whole dataset

# In[41]:


def predictBelegung(df):
    features = df[featurevector].values
    prediction = int(classifier.predict([features]))
    return prediction


# Fire it on the whole Dataset

# In[42]:


data['Vorhersage'] = data.apply(predictBelegung, axis=1)


# In[43]:


data.head(5)


# In[44]:


data['Vorhersage'] = data['Vorhersage'].rolling(window=8).mean().shift(-4)


# In[69]:


plotbelegung(data[['Belegung', 'Vorhersage']], name, '2016-12-10', '2016-12-30')


# In[46]:


data['Vorhersage']['2016-05-14'].plot()


# # If we want to predict the future, let's create it

# In[61]:


future = pd.DataFrame(index=pd.date_range('2017-12-01', '2019-01-01', freq='15Min'))
future.index.name = 'date'
future['Wochentag'] = future.index.dayofweek
#future['KW'] = future.index.week
future['Uhrzeit'] = future.index.hour*60.0 + future.index.minute
ferienseries = pd.Series(future.index, name='Schulferien', index=future.index).apply(isschulferien)
future['Schulferien'] = ferienseries
feiertagseries = pd.Series(future.index, name='Feiertage', index=future.index).apply(shoppingdaystonextfeiertag)
future['bisFeiertag'] = feiertagseries
feiertagseries = pd.Series(future.index, name='Feiertage', index=future.index).apply(shoppingdaysafterfeiertag)
future['nachFeiertag'] = feiertagseries
sonntagsseries = pd.Series(future.index, name='offeneSonntage', index=future.index).apply(isoffenersonntag)
future['offenerSonntag'] = sonntagsseries
weihnachtsseries = pd.Series(future.index, name='Weihnachten', index=future.index).apply(isweihnachten)
future['Weihnachten'] = weihnachtsseries


# ### And predict the occupancy of the parking-space `Centrum-Galerie` with the future features

# In[62]:


future['Vorhersage'] = future.apply(predictBelegung, axis=1)


# In[63]:


future['Vorhersage'] = future['Vorhersage'].rolling(window=8).mean().shift(-4)


# In[64]:


future.head()


# In[65]:


plotbelegung(future['Vorhersage'], name, '2017-12-1', '2017-12-31')
plt.savefig('%s-Belegung-Vorhersage-2018.png' % name, bbox_inches='tight', dpi=150)


# ### Save as .csv

# In[68]:


future.dropna(inplace=True)
future['Vorhersage'].to_csv('%s-Belegung-Vorhersage-2018-15min.csv' % name,
                            header=False,
                            float_format='%i',
                            date_format ='%Y-%m-%dT%H:%M:%S')


# ## Save the Classifier for later use
# 
# Later you can load back the pickled model (possibly in another Python process) with:
# 
# `with open('classifier.pkl', 'rb') as fid:
#     classifier = pickle.load(fid)`

# In[67]:


import pickle
with open('./Classifier/DecisionTreeRegressor-%s.pkl' % name, 'wb') as fid:
    pickle.dump(classifier, fid)  


# Questions: [@Balzer82](https://twitter.com/Balzer82)
