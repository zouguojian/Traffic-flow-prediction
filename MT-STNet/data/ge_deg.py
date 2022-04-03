# -- coding: utf-8 --
import pandas as pd
import csv
site_num=66
out_deg=pd.read_csv('adjacent_direction.csv',encoding='utf-8')
full_deg=pd.read_csv('adjacent_fully.csv',encoding='utf-8')
print(out_deg)
print(full_deg)

out_deg_dict={i:0 for i in range(site_num)}
full_deg_dict={i:0 for i in range(site_num)}
in_deg_dict={i:0 for i in range(site_num)}

for line in full_deg.values:
    full_deg_dict[line[0]]+=1

for line in out_deg.values:
    out_deg_dict[line[0]] += 1

for i in range(site_num):
    in_deg_dict[i]=full_deg_dict[i]-out_deg_dict[i]

file = open('out_deg.csv', 'w', encoding='utf-8')
writer = csv.writer(file)
writer.writerow(['index','out_deg'])
for i in range(site_num):
    writer.writerow([i,out_deg_dict[i]])
file.close()

file = open('in_deg.csv', 'w', encoding='utf-8')
writer = csv.writer(file)
writer.writerow(['index','in_deg'])
for i in range(site_num):
    writer.writerow([i,in_deg_dict[i]])
file.close()