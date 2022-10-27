import json

with open("pred_results.json","r",encoding="utf-8")as result:
    print(type(result))   # <class '_io.TextIOWrapper'>
    data = json.load(result)
    print(type(data))   #  <class 'list'>
    # print("data[0]是:")
    # print("{}".format(data[0]))
    # print("文件名：{}".format(data[0][0]))

    # zheng = []
    # zheng.append(int(data[0][1][len(data[0][1])-1][0]))
    # zheng.append(float(format(data[0][1][len(data[0][1])-1][1],'.3f')))
    # zheng.extend(list(map(int, data[0][1][len(data[0][1])-1][2:6])))
    # print(data[0][1][len(data[0][1])-1])
    # print(zheng)
    print("---------------------------------------------------")
    #遍历0-5
    with open("pred_result.txt","w",encoding="utf-8")as txt:
        for i,its in enumerate(data):
            f_name = its[0]
            for c,clas in enumerate(its[1]):
                it = [f_name]
                it.append(int(clas[0]))
                it.append(float(format(clas[1],'.3f')))
                it.extend(list(map(int, clas[2:6])))
                txt.writelines(it[0]+" "+str(it[2])+" "+str(it[3])+" "+str(it[4])+" "+str(it[5])+" "+str(it[6])+" "+str(it[1])+"\n")
                print("成功写入：{}".format(it))
                
