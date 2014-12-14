import os
import re
import glob
from types import IntType, LongType, FloatType

# checking if a string is a number 
def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# return all of the float numbers of a file.
def grepFloatNumbers(data_filename):
    data_file = open(data_filename)
    data_str = data_file.read()
    data_str = re.split(',| |\t|\n|\f|\r|\[|\]|\(|\)',data_str)
    data = []
    for ele in data_str:
        if isNumber(ele):
            data.append(float(ele))
    return data

def grepStrWithKey(data_filename,key):
    data_file = open(data_filename)
    data_lines = data_file.readlines()
    data = []
    for line in data_lines:
        if line.find(key) >= 0:
            data.append(line[len(key):-1])
    return data

# search each line of the file, return the float number related to the key
# words.
# example of the file exp.txt:
# f(x) = 100.0
# p(x) = 90.0
# f(x) = 70.0
# then grepNumberWithKey("exp.txt","f(x)") will return [100,70],
# and grepNumberWithKey("exp.txt","p(x)") will return [90].
def grepNumberWithKey(data_filename,key,use_number_pos=False,number_position=-1):
    data_file = open(data_filename)
    data_lines = data_file.readlines()
    data = []
    for line in data_lines:
        if line.find(key) >= 0:
            eles = line.split()
            if not use_number_pos:
                for ele in eles:
                    if isNumber(ele):
                        data.append(float(ele))
            elif abs(number_position) < len(eles) and isNumber(eles[number_position]):
                data.append(float(eles[number_position]))
    return data

# return a list of number list with respect to a a list of keywords: each number
# list correspond to one keyword.
def grepNumbersWithKeys(log_filename,keywords):
    numbers_lists = []  
    for key in keywords:
        numbers_lists.append(grepNumberWithKey(log_filename,key))
    return numbers_lists

# change elements of a file
def changeElements(filename,ele_name,ele_value):
    fread = open(filename,"r")
    contents = fread.read()
    contents = contents.replace(ele_name,ele_value)
    fread.close()
    
    fwrite = open(filename,"w")
    fwrite.write(contents)
    fwrite.close()

def average(number_list):
    av = 0
    if len(number_list) > 0:
        av = ( sum(number_list)/len(number_list) )
    return av

# make the document, where doc_dir is the directory include the Makefile
def make_doc(doc_dir):
    os.chdir(doc_dir)
    os.system("make 4 -j8")

def make_doc_one_pass(doc_dir):
    os.chdir(doc_dir)
    os.system("make -j8")

# return the all the path of the files with the given appendix in a directory.
def getFilepath(directory, appendix):
    filepath = []
    os.chdir(directory)
    for fname in glob.glob("*."+appendix):
        filepath.append( os.path.join(directory,fname) )
    return filepath

# make a directory if it is not existed
def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def replace_file_by_key(file_const, file_changed, save_to, key_word):
    """
    replace the lines in file_changed containts the key_word with the
    corresponding contents in file_const, save the results to the file 'save_to'.
    """

    file_1 = open(file_const, "r").readlines()
    file_2 = open(file_changed, "r").readlines()
    
    key_values = []
    for line in file_1:
        if line.find(key_word) >= 0:
            key_values.append(line)

    result_con = []
    k = 0
    for line in file_2:
        if line.find(key_word) < 0:
            result_con.append(line)
        elif k < len(key_values):
            result_con.append(key_values[k])
            k += 1
        else:
            break;

    print "write " + save_to
    file_save_to = open(save_to, 'w')
    for line in result_con:
        file_save_to.write(line)


# scaled the values's list into [0,1].
def unitScale(values):
    max_val = max(values)
    min_val = min(values)
    scaled_val = []
    for v in values:
        new_v = (v-min_val)/(max_val-min_val)
        scaled_val.append(new_v)
    return scaled_val
