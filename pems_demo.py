'''
Created on Jul 20, 2016

@author: wuchunlei
'''

import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
from keras.models import Sequential
from keras.layers import recurrent
from keras.layers.core import Dense, Dropout, Activation
from keras.engine.training import slice_X

FENG_MINUTE = 5
ROLLING = 5
OUTPUT_COUNT = 2
HIDDEN_DIM = 256

def transform(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)
    
def inverse_transform(x, xmin, xmax):
    return (xmax - xmin) * x + xmin
    

# fix random seed for reproducibility
print '------------------------------------------------------------------'
np.random.seed(2016)
RNN = recurrent.SimpleRNN
    
def read_df():
    strindex = str(1).zfill(2)
    file_name = '~/Documents/ml_data/' + 'corridor_2002_2015_' + \
                strindex + '/2002_pri_2015_' + strindex + '.xls'
    output_df = pd.read_excel(file_name, sheetname='Speed', parse_cols="A:B")
    col_1 = output_df.columns[1]
    index = 2
    while index <= 12:
        strindex = str(index).zfill(2)
        file_name = '~/Documents/ml_data/' + 'corridor_2002_2015_' + \
                    strindex + '/2002_pri_2015_' + strindex + '.xls'
        source_df = pd.read_excel(file_name, sheetname='Speed', parse_cols="A:B")
        temp_col = source_df.columns[1]
        if temp_col == col_1:
            output_df = output_df.append(source_df)
        index = index + 1
    return output_df

print 'read data from excel.'
source_df = read_df()
col_0 = source_df.columns[0]
col_1 = source_df.columns[1]
# drop speed nan
source_df = source_df.drop(source_df[source_df[col_1].isnull() == True].index)
dateformat = "%m/%d/%Y %H:%M"
source_df[col_0] = pd.to_datetime(source_df[col_0], errors='coerce', format=dateformat)
#drop timestamp nan
source_df = source_df.drop(source_df[source_df[col_0].isnull() == True].index)
# reindex
#source_df.index = range(len(source_df.index))
# normalization
fun_args = (source_df[col_1].min(), source_df[col_1].max())
source_df[col_1] = source_df[col_1].apply(transform, args=fun_args)

def separate_data(datasets, rolling, ncount):
    col_0 = datasets.columns[0]
    col_1 = datasets.columns[1]
    datemin = datasets[col_0].min()
    datemax = datasets[col_0].max()
    start_date = datemin
    end_date = datemax
    docX, docY, index_date = [], [], []
    print 'start_date:', start_date
    print 'end_date:', end_date
    start_month = start_date.month
    while start_date < end_date:
        next_date = start_date + datetime.timedelta(minutes = FENG_MINUTE*rolling)
        result_date = next_date + datetime.timedelta(minutes = FENG_MINUTE*ncount)
        if result_date.month != start_month:
            print 'processing month:', result_date.month
            start_month = result_date.month
        # only select 06:00~22:00
        start_slice = start_date.strftime('%H%M')
        if start_slice < '0600' or start_slice >= '2130':
            start_date = start_date + datetime.timedelta(minutes = FENG_MINUTE)
            continue
        train_df = datasets[(datasets[col_0] >= start_date) & (datasets[col_0] < next_date)]
        result_df = datasets[(datasets[col_0] >= next_date) & (datasets[col_0] < result_date)]
        if len(train_df) != rolling or len(result_df) != ncount:
            start_date = start_date + datetime.timedelta(minutes = FENG_MINUTE)
            continue
        docX.append(train_df[col_1].as_matrix())
        docY.append(result_df[col_1].as_matrix())
        index_date.append(start_date + datetime.timedelta(minutes = FENG_MINUTE*(rolling-1)))
        start_date = start_date + datetime.timedelta(minutes = FENG_MINUTE)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY, index_date

print 'separate data.'
X_data, y_data, index_date = separate_data(source_df, ROLLING, OUTPUT_COUNT)

# shuffle
#indices = np.arange(len(X_data))
#np.random.shuffle(indices)
#X_data = X_data[indices]
#y_data = y_data[indices]
# split data
split_at = len(X_data) - len(X_data) / 10
(X_train, x_test) = (slice_X(X_data, 0, split_at), slice_X(X_data, split_at))
(Y_train, y_test) = (slice_X(y_data, 0, split_at), slice_X(y_data, split_at))
# reshape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print X_train.shape
print y_test.shape
# model
print 'model compile.'
model = Sequential()
model.add(RNN(HIDDEN_DIM, input_dim=ROLLING))
model.add(Dense(OUTPUT_COUNT))
# model compile
model.compile(loss='mean_squared_error', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

# train the model
BATCH_SIZE = 128
NB_EPOCH = 20
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
          nb_epoch=NB_EPOCH, verbose=1, 
          validation_data=(x_test, y_test))

train_predict = model.predict(X_train)
test_predict = model.predict(x_test)

# plot the datasets
df = pd.DataFrame(X_data[:, -1].copy(), index=index_date, columns=[col_1])
df[col_1] = df[col_1].apply(inverse_transform, args=fun_args)
sts_df = df.copy()
ncount = 0
all_predict = np.append(train_predict, test_predict, axis=0)
while ncount < OUTPUT_COUNT:
    col_2 = str(col_1) + '_' + str(ncount)
    df[col_2] = all_predict[:, ncount]
    df[col_2] = df[col_2].apply(inverse_transform, args=fun_args)
    ncount = ncount + 1

all_source = np.append(Y_train, y_test, axis = 0)
print 'all_source:', all_source.shape
print 'all_predict:', all_predict.shape
print 'statistic:'
ncount = 0
while ncount < OUTPUT_COUNT:
    '''
    5%, 10%, 15%
    '''
    col_2 = str(col_1) + '_' + str(ncount)
    sts_df[col_2] = all_source[:, ncount]
    sts_df[col_2] = sts_df[col_2].apply(inverse_transform, args=fun_args)
    rate = (df[col_2].values - sts_df[col_2].values) / sts_df[col_2].values * 100.0
    print 'rate:', ncount
    print '(,-15%]      :', sum(rate <= -15.0) * 1.0 / len(all_source) * 100.0
    print '(-15%, -10%] :', sum((-15.0 < rate) & (rate <= -10.0)) * 1.0 / len(all_source) * 100.0
    print '(-10%, -5%]  :', sum((-10.0 < rate) & (rate <= -5.0)) * 1.0 / len(all_source) * 100.0
    print '(-5%, 5%)    :', sum((-5.0 < rate) & (rate < 5.0)) * 1.0 / len(all_source) * 100.0
    print '[5%, 10%)    :', sum((5.0 <= rate) & (rate < 10.0)) * 1.0 / len(all_source) * 100.0
    print '[10%, 15%)   :', sum((10.0 <= rate) & (rate < 15.0)) * 1.0 / len(all_source) * 100.0
    print '[15%, )      :', sum(15.0 <= rate) * 1.0 / len(all_source) * 100.0
    ncount = ncount + 1

first_date = df.index.min()
first_date = datetime.datetime(first_date.year, first_date.month, first_date.day)
last_date = df.index.max()
# plot all sub plots
print 'plot all data.'
while first_date < last_date:
    stop_date = datetime.datetime(first_date.year + first_date.month / 12, 
                                  (first_date.month % 12 + 1), 1)
    month_df = df[(df.index >= first_date) & (df.index < stop_date)]
    fig = plt.figure(dpi=300)
    date_counts = (stop_date - first_date).days + 1
    fig.set_size_inches(30, 5 * date_counts)
    file_name = '/home/wuchunlei/Documents/ml_data/all_plot/' + first_date.strftime('%Y_%m') + '.png'
    nindex = 1
    while nindex <= date_counts:
        ax = fig.add_subplot(date_counts, 1, nindex)
        end_date = first_date + datetime.timedelta(days=1)
        plt_df = month_df[(month_df.index >= first_date) & (month_df.index < end_date)]
        ax.plot(plt_df.index, plt_df[col_1], color='g', linewidth=1.5)
        title = first_date.strftime("%Y%m%d") + ' ' + first_date.strftime("%A")
        ax.set_title(title)
        for i in range(len(plt_df.index) - OUTPUT_COUNT):
            plt_data = plt_df.ix[plt_df.index[i]].tolist()
            ax.plot(plt_df.index[i:i+len(plt_data)].tolist(), plt_data, color='k', alpha=0.4, linewidth=0.8)
        first_date = end_date
        nindex = nindex + 1
    fig.savefig(file_name)
    first_date = stop_date
    fig.clf()
plt.close()
print '------------------------------------------------------------------'
