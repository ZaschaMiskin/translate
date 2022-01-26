import bpe
import data

text = data.read_txt("text.txt")
bpe_ops = bpe.learn_bpe(text, 100)
bpe_text = bpe.bpe(text, bpe_ops)
