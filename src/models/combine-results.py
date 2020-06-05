import pandas as pd
import glob
import numpy as np
import os

root_path = './'
tasks = next(os.walk(root_path))[1]
tasks.sort()

all_tasks = ['german', 'adult', 'bank', 'home', 'titanic']

for task in all_tasks:
    base_path = root_path + task + '/res/'

    # For each model, compute mean of all the experiments and put into a single file
    m_list = next(os.walk(base_path))[1]
    all_li = []

    m_list.sort()
    for model in m_list:
        path = base_path + model
        all_files = glob.glob(path + "/*.csv")
        li = []
        #print('Loading csv files in ' + path)
        for filename in all_files:
            try:
                df = pd.read_csv(filename, index_col=None, header=0)
            except:
                print('Error loading raw-results. Check the file: ' + filename)
            #del df['GTI']
            df['DI'] = np.log(df['DI'])
            df['CNT'] = np.log(df['CNT'])
            li.append(df)

        df_concat = pd.concat(li)
        by_row_index = df_concat.groupby(df_concat.index)
        avg = by_row_index.mean()

        avg = avg.round(8)
        avg.replace([np.inf, -np.inf], np.nan, inplace=True)
        avg.fillna(avg.iloc[0], inplace=True)

        #print('Combining experiments. Writing ' + base_path + model + '-avg.csv')
        avg.to_csv(base_path + model + '-avg.csv', index=False)

    # For each task, combine results of all the models and combine into a single file.
    all_files = glob.glob(base_path + "/*.csv")
    li = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

    all_files.sort()
    for filename in all_files:
        for i in range(8):
            df = pd.read_csv(filename, index_col=None, header=0)
            df  = df.loc[[i]]
            li[i].append(df)

    all_mdl = []
    for i in range(8):
        all_m = pd.concat(li[i])
        all_m.index = range(8)
        all_mdl.append(all_m.round(3))
    final = pd.concat(all_mdl, axis = 1)
    final.insert(0, 'Model', range(1, 9))
    final.insert(10, 'RW', [''] * 8)
    final.insert(20, 'DIR', [''] * 8)
    final.insert(30, 'AD', [''] * 8)
    final.insert(40, 'PR', [''] * 8)
    final.insert(50, 'EO', [''] * 8)
    final.insert(60, 'CEO', [''] * 8)
    final.insert(70, 'ROC', [''] * 8)
    print('Combining results for all the models. Writing to ' + base_path + '../' + 'all_model.csv')
    final.to_csv(base_path + '../' + 'all_model.csv', index=False)

all_res = []
for task in all_tasks:
    path = root_path + task + '/all_model.csv'
    df = pd.read_csv(path, index_col=None, header=None)

    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header

    all_res.append(df)

combined = pd.concat(all_res)
combined.insert(0, 'Dataset', [all_tasks[0]] + [''] * 7 + [all_tasks[1]] + [''] * 7 + [all_tasks[2]] + [''] * 7 + [all_tasks[3]] + [''] * 7 + [all_tasks[4]] + [''] * 7)
print('Combining results for all the tasks. Writing to ' + root_path + 'final.csv')
combined.to_csv(root_path + 'final.csv', index=False)
