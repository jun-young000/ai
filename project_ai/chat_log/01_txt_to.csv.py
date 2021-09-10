import os
import re


name_list=os.listdir("tmp/")
# print(name_list)


if not os.path.exists("data"):
    os.mkdir("data")

for name in name_list:
    with open("tmp/"+name, "r", encoding="UTF8") as file:
        to_csv=list()

        while True:
            line=file.readline()
            if not line:break


            # print(line)

            tmp=line.split(" ")
            # print(tmp)

        to_csv.append(tmp[0]+","+re.sub('\d-|\([^dm].*\)|\d', '',tmp[2])+","+re.sub('\d-|\([^dm]\)|\d','',tmp[4])+","+tmp[5])

    # print(to_csv)
    with open("data/"+name.split(".")[0]+".csv","w",encoding="UTF8") as csv:
        for data in to_csv:
            csv.write(data)

