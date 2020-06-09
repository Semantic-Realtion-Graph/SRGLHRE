import json

def convert(inputfile,outputfile):
    file = open(inputfile,'r')
    dicts = json.load(file)
    out = open(outputfile,'w')
    for dic in dicts:
        new_dict = dict()
        tokens = dic["token"]
        new_dict["tokens"] = tokens
        new_dict["label"] = dic["relation"]

        subject = []
        subject.append(dic["subj_start"])
        subject.append(dic["subj_end"]+1)
        object = []
        object.append(dic["obj_start"])
        object.append(dic["obj_end"] + 1)
        position = []
        position.append(subject)
        position.append(object)
        new_dict["entities"]=position

        new_dict_str = json.dumps(new_dict)
        out.write(new_dict_str)
        out.write("\n")




if __name__ == '__main__':
    convert("train.json","train.jsonl")