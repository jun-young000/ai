import os
import re


name_list= os.listdir("data/")
file_names=list(map(lambda  x:x.slpit(".")[0],name_list))
# print(file_names)

result=list()
result.append("chat_data,chat_from,chat_to,chat_content\n")
for i, name in enumerate(name_list):
    with open("data/"+name, "r",encoding="UTF8") as file:
        while True:
            line=file.readline()
            if not line:break
            result.append(file_names[i]+","+line)

# print(result)


with open("data/all.csv","w",encoding="UTF8") as csv:
    for data in result:
        csv.write(data)