import os
import re
import json
from extract_label import extract_label
import pickle
import sys
sys.setrecursionlimit(3000)

def getBiasLabels(dir_name):
    file_names = os.listdir(dir_name)
    # file_names=['BuzzFeed_Fake_1-Webpage.json']
    list_=[]
    bias_data={}
    for file_name in file_names:
        source  = ''
        label   = ''
        # print file_name
        with open(dir_name+file_name) as f:
            Json_dict=json.load(f)
        # text_data=Json_dict['text']
        # SourcePattern=re.compile(u'Source:\s*([\w ]+)',re.IGNORECASE)
        # AllSources=re.findall(SourcePattern,text_data)
        # for source in AllSources:
        #     print source
        if('source' in Json_dict.keys()):
            # print Json_dict[u"source"]
            source,label = extract_label(Json_dict[u"source"])
        else:
            list_.append(file_name)
        
        print file_name, source,label
        bias_data[file_name]={'source':source,'label':label}

    return bias_data

def main():
    dir_name='./Data/PolitiFact/FakeNewsContent/'
    bias_data=getBiasLabels(dir_name)
    # print bias_data
    f = open('PolitiFact_Fake_Bias.pkl','w')
    pickle.dump(bias_data,f)
    f.close()

if __name__ == '__main__':
    main()
