from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import UnicodeScripts
from tokenizers.trainers import BpeTrainer
import sys

txt_doc = sys.argv[1]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = UnicodeScripts()

trainer = BpeTrainer(special_tokens = ["[PAD]","[UNK]","[CLS]","<kan_start>","<tel_start>","<tam_start>","<hin_start>","<kan_end>","<tel_end>","<tam_end>","<hin_end>"])

tokenizer.train([txt_doc],trainer)
tokenizer.save("./tokenizer")

print("The Tokenizer is Saved")