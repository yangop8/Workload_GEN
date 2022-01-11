import os
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import pickle

files = ['books','fb','osm_cellids','wiki_ts']
color = ['#0076ae', '#ff7400', '#00a13b', '#ef0000', '#985247','#fdb813','#88aca1','#6534ac','#d4c99e','#76daff','#444444','#ee70a6']
distribution_type = ['zipf','normal','poisson','pareto','binomial']
INTERVAL = 50

def read_lookup_data(type:str):
    lookups = {}
    for file in files:
        path = './data/'+file+'_200M_uint64_equality_lookups_'+type
        fin = open(path,'rb')

        result = []

        size = fin.read(8)
        num = struct.unpack('Q',size)
        size_num = num[0]

        size_total = os.path.getsize(path)

        for i in range(size_num):
            data = fin.read(16)
            num1 = struct.unpack('Q',data[:8])
            num2 = struct.unpack('Q',data[8:])
            result.append((num1[0],num2[0]))
        
        lookups[file] = result

        fin.close()
    return lookups

def read_raw_data():
    raw_data = {}
    for file in files:
        print ('Current file is: '+file)
        path = './data/'+file+'_200M_uint64'
        fin = open(path,'rb')

        result = []

        size = fin.read(8)
        num = struct.unpack('Q',size)
        size_num = num[0]

        size_total = os.path.getsize(path)

        for i in tqdm(range(size_num)):
            data = fin.read(8)
            num1 = struct.unpack('Q',data)
            result.append(num1[0])
        
        raw_data[file] = result

        fin.close()
    return raw_data

def read_workload_data(type:str, slot:int):
    workloads = {}
    for file in files:
        path = './synthetic/'+file+'_200M_uint64_same_dis_lookups_1M_'+type
        fin = open(path,'rb')

        result = []

        size = fin.read(8)
        num = struct.unpack('Q',size)
        size_num = num[0]

        size_total = os.path.getsize(path)
        for j in range(slot):
            tmp_result = []
            for i in range(int(size_num/slot)):
                data = fin.read(16)
                num1 = struct.unpack('Q',data[:8])
                num2 = struct.unpack('Q',data[8:])
                tmp_result.append(num1[0])
            result.append(tmp_result)
        
        workloads[file] = result

        fin.close()
    return workloads

def group(keys:list, k:int, section:list=None):
    l = len(keys)

    if section == None or len(section) != k:
        max_pos = l-1
        interval = int((keys[max_pos]-keys[0])/k)
        while (keys[max_pos]-keys[max_pos-1] > interval/10):
            max_pos -= 1
            interval = int((keys[max_pos]-keys[0])/k)
        section = []
        for i in range(k-1):
            section.append([keys[0]+i*interval,keys[0]+(i+1)*interval])
        section.append([keys[0]+(k-1)*interval,keys[l-1]+1])
    else:
        section.pop()
        section.append([section[-1][1]+1, max(keys)+1])

    divided_list=[[] for i in range(k)]
    for i in tqdm(range(l)):
        for j in range(k):
            if section[j][0] <= keys[i] < section[j][1]:
                divided_list[j].append(keys[i])
    divided_num = [len(i) for i in divided_list]
    return section, divided_list, divided_num

def bar_plot(file:str,section:list,divided_num:list,divided_num_lookup_1M:list, divided_num_lookup_10M:list):
    fig = plt.figure(figsize=(27,18))
    plt.suptitle(file, y=0.93, fontsize=30)
    ax1 = plt.subplot(2,3,1)
    color_raw = []
    for i in divided_num:
        if i < np.mean(divided_num):
            color_raw.append(color[0])
        else:
            color_raw.append(color[1])
    ax1.bar(range(len(section)),divided_num,color=color_raw)
    plt.gca().set_title(file+'_raw_data_distribution')
    ax2 = plt.subplot(2,3,2)
    color_raw_lookup = []
    for i in divided_num_lookup_1M:
        if i < np.mean(divided_num_lookup_1M):
            color_raw_lookup.append(color[0])
        else:
            color_raw_lookup.append(color[1])
    ax2.bar(range(len(section)),divided_num_lookup_1M,color=color_raw_lookup)
    plt.gca().set_title(file+'_1M_lookup_distribution')
    ax3 = plt.subplot(2,3,3)
    color_raw_lookup = []
    for i in divided_num_lookup_10M:
        if i < np.mean(divided_num_lookup_10M):
            color_raw_lookup.append(color[0])
        else:
            color_raw_lookup.append(color[1])
    ax3.bar(range(len(section)),divided_num_lookup_10M,color=color_raw_lookup)
    plt.gca().set_title(file+'_10M_lookup_distribution')
    ax4 = plt.subplot(2,3,4)
    cdf_num = [sum(divided_num[:i])  for i in range(len(divided_num))]
    ax4.plot(range(len(section)),cdf_num,color=color[0])
    plt.gca().set_title(file+'_raw_data_CDF')
    ax5 = plt.subplot(2,3,5)
    cdf_num_lookup = [sum(divided_num_lookup_1M[:i]) for i in range(len(divided_num_lookup_1M))]
    ax5.plot(range(len(section)),cdf_num_lookup,color=color[0])
    plt.gca().set_title(file+'_1M_lookup_CDF')
    ax6 = plt.subplot(2,3,6)
    cdf_num_lookup = [sum(divided_num_lookup_10M[:i]) for i in range(len(divided_num_lookup_10M))]
    ax6.plot(range(len(section)),cdf_num_lookup,color=color[0])
    plt.gca().set_title(file+'_10M_lookup_CDF')
        
    plt.savefig('./plot/'+file+'.png',dpi=150)

def bar_plot_workload(file:str,section:list,divided_num:list,divided_num_workload:list, distribution_type:str):
    fig = plt.figure(figsize=(27,18))
    plt.suptitle(file, y=0.93, fontsize=30)
    ax1 = plt.subplot(2,2,1)
    color_raw = []
    for i in divided_num:
        if i < np.mean(divided_num):
            color_raw.append(color[0])
        else:
            color_raw.append(color[1])
    ax1.bar(range(len(section)),divided_num,color=color_raw)
    plt.gca().set_title(file+'_raw_data_distribution')
    ax2 = plt.subplot(2,2,2)
    color_raw_workload = []
    total_divided_workload = []
    for divided_workload in divided_num_workload:
        for i in divided_workload:
            if i < np.mean(divided_workload):
                color_raw_workload.append(color[0])
            else:
                color_raw_workload.append(color[1])
            total_divided_workload.append(i)
    ax2.bar(range(len(total_divided_workload)),total_divided_workload,color=color_raw_workload)
    plt.gca().set_title(file+'_workload_distribution')

    ax3 = plt.subplot(2,2,3)
    cdf_num = [sum(divided_num[:i])  for i in range(len(divided_num))]
    ax3.plot(range(len(section)),cdf_num,color=color[0])
    plt.gca().set_title(file+'_raw_data_CDF')
    ax4 = plt.subplot(2,2,4)
    divided_num_workload_sum = np.sum(divided_num_workload,axis=0)
    cdf_num_workload = [sum(divided_num_workload_sum[:i]) for i in range(len(section))]
    ax4.plot(range(len(section)),cdf_num_workload,color=color[0])
    plt.gca().set_title(file+'_workload_CDF')

        
    plt.savefig('./plot/'+file+'_'+distribution_type+'.png',dpi=150)

def check(a,b):
    for (key,value) in b:
        print (a.index(key), value)

def save_pickle(file:str, section:list, divided_num:list):
    fout = open('./plot/'+file+'_pickle.pkl','wb')
    input = [section,divided_num]
    pickle.dump(input, fout)
    fout.close()

def main():
    print ('--------LOADING---------')    
    raw_data = read_raw_data()
    lookup_1M = read_lookup_data('1M')
    lookup_10M = read_lookup_data('10M')
    workload = read_workload_data(distribution_type[1],10)
    print ('--------GROUPBY---------')
    for file in files:
        print ('Current file is: '+file)
        keys = raw_data[file]
        keys_lookup_1M = [x[0] for x in lookup_1M[file]]
        keys_lookup_10M = [x[0] for x in lookup_10M[file]]
        keys_workload = workload[file]
        
        pickle_file = file+'_pickle.pkl'
        if os.path.exists('./plot/'):
            continue
        else:
            os.mkdir('./plot/')
            
        if file == 'fb':
            section, divided_list_lookup_10M, divided_num_lookup_10M = group(keys_lookup_10M, INTERVAL)
        if pickle_file in os.listdir('./plot/'):
            fin = open(pickle_file,'rb')
            fin_content = pickle.load(fin)
            section, divided_num = fin_content[0], fin_content[1]
            fin.close()
        else:
            if file == 'fb':
                section, divided_list, divided_num = group(keys, INTERVAL, section)
            else:
                section, divided_list, divided_num = group(keys, INTERVAL)
            save_pickle(file, section, divided_num)
        section, divided_list_lookup_10M, divided_num_lookup_10M = group(keys_lookup_10M, INTERVAL, section)
        section, divided_list_lookup_1M, divided_num_lookup_1M = group(keys_lookup_1M, INTERVAL, section)
        
        divided_num_workload = []
        for i in range(len(keys_workload)):
            section, divided_list_workload, divided_num_workload_slot = group(keys_workload[i], INTERVAL, section)
            divided_num_workload.append(divided_num_workload_slot)
        bar_plot(file, section, divided_num, divided_num_lookup_1M, divided_num_lookup_10M)
        bar_plot_workload(file, section, divided_num, divided_num_workload, distribution_type[1])
        

if __name__ == '__main__':
    main()