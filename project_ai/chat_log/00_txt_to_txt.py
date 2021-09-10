import os
import re

name_list = os.listdir("log/")
file_names = list(map(lambda x: x.split(".")[0], name_list))
# print(file_names)

if not os.path.exists("tmp"):
    os.mkdir("tmp")

for file_name in file_names:
    txt = ""
    with open("log/" + file_name + ".txt", encoding="UTF8") as file:
        while True:
            line = file.readline()
            if not line: break
            line = line.replace("Minjeong Kang", "강민정")\
                        .replace("(매니저) 민채원", "멀티캠퍼스_매니저")\
                        .replace("멀티캠퍼스_민채원", "멀티캠퍼스_매니저")
            txt += line
    # print(txt)

    time = re.findall("\d{2}:\d{2}:\d{2} ", txt)
    split_txt = re.split("\d{2}:\d{2}:\d{2} ", txt)[1:]

    result = list()
    for i in range(len(time)):
        result.append(time[i] + split_txt[i].replace(split_txt[i][split_txt[i].index(":"):], " 내용"))
    # print(result)

    with open("tmp/" + file_name + ".txt", "w", encoding="UTF8") as file:
        for line in result:
            file.write(line+"\n")
