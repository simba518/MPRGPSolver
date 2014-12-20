#! /usr/bin/env python
import os
from utility import *
import matplotlib.pyplot as plt

def grep_data(data_filename, key, split_label):

    data_file = open(data_filename)
    data_lines = data_file.readlines()
    all_data = []
    data = []
    for line in data_lines:
        if line.find(key) >= 0:
            d = float(line.split()[-1])
            data.append(d)
        elif line.find(split_label) >= 0:
            all_data.append(data)
            data = []
    return all_data

def grep_and_save_fig(log_file, key, split_label, fig_name):
    
    all_data = grep_data(log_file, key, split_label)
    for data in all_data:
        plt.plot(range(0,len(data)), data)
    plt.savefig(fig_name)
    plt.clf()
    return fig_name

#-----------------------------main------------------------------------------------
report_tex = "./tempt/report_func.tex"
log_fs = os.listdir("./tempt")
tex_str = open("./script/report_head.tex").read()

for log_f in log_fs:

    if not log_f.endswith(".txt"): continue
    print log_f

    tempt = report_tex+"tempt"
    os.system("cp "+"./script/report_data_func.tex "+tempt)
    log_f = "./tempt/"+log_f

    fig = grep_and_save_fig(log_f,"func = ","monotonic mprgp solving:",log_f[0:-4]+"func.png")
    changeElements(tempt,"#func-curves#",fig.replace("_","\\string_"))
    changeElements(tempt,"#log-file#", log_f.split()[-1].replace("_","\\_") )

    log_f = "./tempt/"+log_f
    tex_str += open(tempt).read()

open(report_tex,'w').write(tex_str+"\n\end{document}")
os.system("pdflatex "+" -output-directory="+os.path.dirname(report_tex)+" "+report_tex)
