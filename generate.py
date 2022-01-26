from numpy import random, square
import numpy as np
import struct
from tqdm import tqdm
import os
import pandas as pd
from multiprocessing import Pool
import math

np.random.seed(1)
files = ['books','fb','osm_cellids','wiki_ts']
# files = ['osm_cellids'] # for test
slot_num = 50
distribution_type = ['zipf','normal','poisson','pareto','binomial']

x = list(range(slot_num))
random.shuffle(x)

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

def spec_rand(type:str):
    if type == distribution_type[0]:
        y = random.zipf(a=2,size=10000)
        y_rand = [x[i-1] for i in y[y<=slot_num]]
        return y_rand
    elif type == distribution_type[1]:
        y = random.normal(slot_num/2,math.sqrt(slot_num/2),size=100000)
        y_rand = [x[int(i)] for i in y[(y>0) & (y<slot_num)]]
        return y_rand
    elif type == distribution_type[2]:
        y = random.poisson(lam=slot_num/2,size=100000)
        y_rand = [x[i-1] for i in y[y<=50]]
        return y_rand
    elif type == distribution_type[3]:
        y = random.pareto(a=3,size=100000)
        y_rand = [x[int(i)] for i in y[y<50]]
        return y_rand
    elif type == distribution_type[4]:
        y = random.binomial(n=100,p=0.5,size=100000)
        y_rand = [x[i] for i in y[y<50]]
        return y_rand
    else:
        return None

def workload_gen(type:str, raw_data:list, total:int=2000):
    sum = len(raw_data)
    interval = int (sum / slot_num)
    y = spec_rand(type)
    y_count = pd.value_counts(y)
    result = []
    for i in tqdm(range(total)):
        select = random.choice(y)
        ss = interval * select
        ee = min(sum, interval * (select + 1) -1)
        pos = random.randint(ss,ee)
        result.append((raw_data[pos],pos))
    return result

def time_gen(type:str, raw_data:list, total:int=2000, period:int=10):
    result = []
    for i in tqdm(range(period)):
        random.shuffle(x)
        result += workload_gen(type, raw_data, total)
    return result

def pack_result(result:list, file:str):
    fout = open(file,'wb')

    size_num = len(result)

    fout.write(struct.pack('Q',size_num))

    for i in tqdm(range(size_num)):
        fout.write(struct.pack('2Q',result[i][0],result[i][1]))

    fout.close()


def main():
    print ('--------LOADING---------')
    raw_data = read_raw_data()
    print ('--------GENERATION---------')

    # multi-threads **cause tqdm in a mess**
    p = Pool(4)
    input1 = [distribution_type[0]]*4
    input2 = [raw_data[files[0]],raw_data[files[1]],raw_data[files[2]],raw_data[files[3]]]
    result = p.starmap(time_gen,zip(input1,input2))
    for i in range(len(result)):
        pack_result(result[i],'./synthetic/'+files[i]+'_200M_uint64_same_dis_lookups_20K_'+distribution_type[0])

    # one-thread
    # for file in files:
    #     print ('Current file is: '+file)
    #     keys = raw_data[file]
    #     result = time_gen(distribution_type[1],keys)
    #     pack_result(result,'./synthetic/'+file+'_200M_uint64_same_dis_lookups_1M_'+distribution_type[1])
        # section, divided_list, divided_num = group(keys, slot_num, section)


if __name__ == '__main__':
    main()

