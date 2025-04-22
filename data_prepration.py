from preprocessing import preprocessing_text
import sys


# This code will help to create a simple and one dataset that which makes easy for the forther steps.

"""This both dictnoires are helpful to find the sentence and label with the start and end so that we have two importance:
      1. It will help to work with teacher forcing so that model will learn easly
      2. The model also is able to predict which language is given and what language it need to convert.
 """
start_dict = {"kannada":"<kan_start>","telugu":"<tel_start>","tamil":"<tam_start>","hindi":"<hin_start>"}
end_dict = {"kannada":"<kan_end>","telugu":"<tel_end>","tamil":"<tam_end>","hindi":"<hin_end>"}

sorce_lang = sys.argv[1]
desc_lang = sys.argv[2]

def prepraining_data(sorce_lang,desc_lang):

    # Here we are loading the both Sorce data and destination data

    with open(r"./Language_data/{0}_data.txt".format(sorce_lang),"r",encoding="utf-8") as txt:

        sorce_text = txt.readlines()

    with open(r"./Language_data/{0}_data.txt".format(desc_lang),"r",encoding="utf-8") as txt:

        desc_text = txt.readlines()

    if len(sorce_text) == len(desc_text):

        with open(r"./final_data.txt","a",encoding="utf-8") as text:
            
            for i in range(len(sorce_text)):

                text.write(start_dict[sorce_lang]+" "+preprocessing_text(sorce_text[i].replace("\n",""))+" "+end_dict[sorce_lang]
                           +"\t\t" + start_dict[desc_lang]+" "+preprocessing_text(desc_text[i].replace("\n",""))+" "+end_dict[desc_lang]+"\n")




    #print([start_dict[sorce_lang]+" "+i.replace("\n","")+" "+end_dict[sorce_lang] for i in sorce_text][0:5])
    #print([start_dict[desc_lang]+" "+i.replace("\n","")+" "+end_dict[desc_lang] for i in desc_text][0:5])

    #print(sorce_text[0:5],desc_text[0:5],len(sorce_text),len(desc_text))

prepraining_data(sorce_lang,desc_lang)

