#! /usr/bin/env python
import os
from utility import *
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------functions--------------------------------------------

def grepInt(log_file,keyname,default=""):
    data = grepNumberWithKey(log_file,keyname)
    if len(data) <= 0:
        return default
    return str(int(data[0]))


def grepFloat(log_file,keyname,default=""):
    data = grepNumberWithKey(log_file,keyname)
    if len(data) <= 0:
        return default
    return str('%.5g'%(data[0]))


def grepStr(log_file,keyname,default=""):
    data = grepStrWithKey(log_file,keyname)
    if len(data) <= 0:
        return default
    return str(data[0])

def grepFloatList(log_file,keyname,default=""):
    data = grepNumberWithKey(log_file,keyname)
    if len(data) <= 0:
        return default
    d = ""
    for var in data:
        d = d +str('%.3g'%(var))+", "
    return d

def count(log_file,key):
    data_file = open(log_file)
    data_lines = data_file.readlines()
    num = 0
    for line in data_lines:
        if line.find(key) >= 0:
            num = num+1
    return num

def sum(log_file,key):
    s = 0
    data = grepNumberWithKey(log_file,key)
    for var in data:
        s = s + var
    return str(s)

def sumInt(log_file,key):
    s = 0
    data = grepNumberWithKey(log_file,key)
    for var in data:
        s = s + int(var)
    return str(s)

def averageInt(log_file,key):
    s = 0
    data = grepNumberWithKey(log_file,key)
    for var in data:
        s = s + int(var)
    n = 1;
    if len(data) > 0:
        n = len(data)
    return str(s/n)

def averageFloat(log_file,key):
    s = 0
    data = grepNumberWithKey(log_file,key)
    for var in data:
        s = s + float(var)
    n = 1;
    if len(data) > 0:
        n = len(data)
    return str( '%.3g'%(s/n) )

def minFloat(log_file,key):
    data = grepNumberWithKey(log_file,key)
    if len(data) > 0:
        return str( '%.3g'%min(data) )
    return str(0)

def maxFloat(log_file,key):
    data = grepNumberWithKey(log_file,key)
    if len(data) > 0:
        return str( '%.3g'%max(data) )
    return str(0)

def grep_and_save_fig(log_file,key_list,label_list,fig_name,use_log=False,use_number_pos=False,number_position=-1):
    for i in range(0,len(key_list)):
        data = grepNumberWithKey(log_file,key_list[i],use_number_pos,number_position)
        if len(data) > 0:
            if use_log:
                plt.semilogy()
            plt.plot(range(0,len(data)),data,label=label_list[i])
    legend = plt.legend(bbox_to_anchor=(0,1.01,1,20), loc=8, ncol=8, mode="expand",borderaxespad=-0.5,labelspacing=-0.2)
    plt.savefig(fig_name)
    plt.clf()
    return fig_name

#-----------------------------main------------------------------------------------
report_tex = "./tempt/report.tex"
log_fs = os.listdir("./tempt")
tex_str = open("./script/report_head.tex").read()

for log_f in log_fs:

    if not log_f.endswith(".txt"): continue
    print log_f

    tempt = report_tex+"tempt"
    os.system("cp "+"./script/report_data.tex "+tempt)
    log_f = "./tempt/"+log_f

    fig = grep_and_save_fig(log_f,["power iter:","MPRGP iter: ", "exp steps: ", "cg steps: ", " prop steps: "],
                            ["power","MPRGP", "exp", "cg", "prop"],
                            log_f[0:-4]+"iterations.png",False)

    changeElements(tempt,"#iter_curves#",fig.replace("_","\\string_"))

    fig = grep_and_save_fig(log_f,["constraints:"],["constraints"],log_f[0:-4]+"cons.png")
    changeElements(tempt,"#constraints#",fig.replace("_","\\string_"))

    changeElements(tempt,"#dimension#", grepInt(log_f, "dimension:"))
    changeElements(tempt,"#number_of_frames#", grepInt(log_f, "total frames:"))
    changeElements(tempt,"#mprgp-it#", grepFloat(log_f, "mprgp max it:"))
    changeElements(tempt,"#mprgp-tol#", grepFloat(log_f, "mprgp tol:"))

    changeElements(tempt,"#non-convergent#", str(count(log_f,"MPRGP is not convergent")))
    changeElements(tempt,"#failed-lambda#", str(count(log_f,"lambda = ")))
    changeElements(tempt,"#av-lambda#", str(averageFloat(log_f,"lambda = ")))
    changeElements(tempt,"#max-lambda#", str(minFloat(log_f,"lambda = ")))
    changeElements(tempt,"#failed-lag-grad#", str(count(log_f,"lag_grad.norm():")))
    changeElements(tempt,"#av-grad#", str(averageFloat(log_f,"lag_grad.norm():")))
    changeElements(tempt,"#max-grad#", str(maxFloat(log_f,"lag_grad.norm():")))

    changeElements(tempt,"#total-cg#", sumInt(log_f,"cg steps: "))
    changeElements(tempt,"#total-exp#", sumInt(log_f,"exp steps: "))
    changeElements(tempt,"#total-prop#", sumInt(log_f,"prop steps: "))
    changeElements(tempt,"#total-power#", sumInt(log_f,"power iter: "))
    changeElements(tempt,"#total-cons#", sumInt(log_f,"constraints: "))

    changeElements(tempt,"#av-cg#", averageInt(log_f,"cg steps: "))
    changeElements(tempt,"#av-exp#", averageInt(log_f,"exp steps: "))
    changeElements(tempt,"#av-prop#", averageInt(log_f,"prop steps: "))
    changeElements(tempt,"#av-power#", averageInt(log_f,"power iter: "))
    changeElements(tempt,"#av-cons#", averageInt(log_f,"constraints: "))


    changeElements(tempt,"#total-time#", averageFloat(log_f,"total solving:"))
    changeElements(tempt,"#pre-time#", averageFloat(log_f,"preconditioning setup:"))
    changeElements(tempt,"#power-time#", averageFloat(log_f,"compute spectral radius:"))
    changeElements(tempt,"#mprgp-time#", averageFloat(log_f,"mprgp solving:"))

    initfilename = grepStr(log_f,"init file:")
    if len(initfilename) > 0:
        changeElements(tempt,"#init_file_name#", log_f.split()[-1].replace("_","\\_") )
    else:
        changeElements(tempt,"#init_file_name#", "" )

    log_f = "./tempt/"+log_f
    tex_str += open(tempt).read()

open(report_tex,'w').write(tex_str+"\n\end{document}")
os.system("pdflatex "+" -output-directory="+os.path.dirname(report_tex)+" "+report_tex)
