import re


def preprocessing_text(text):

    text = re.sub(r"\n","",text)
    text = re.sub(r"\u200c","",text)
    text = re.sub(r'"',"",text)
    text = re.sub(r"'","",text)
    text = re.sub(r"\u200d","",text)
    text = re.sub(r"\u200a","",text)
    text = re.sub(r"\u200b","",text)
    text = re.sub(r"\(","",text)
    text = re.sub(r"\)","",text)
    text = re.sub(r"\{","",text)
    text = re.sub(r"\}","",text)
    text = re.sub(r"\xa0"," ",text)
    text = re.sub(r"\[","",text)
    text = re.sub(r"\]","",text)
    text = re.sub(r"\-","",text)
    text = re.sub(r'\""',"",text)
    
    return text