#!/usr/bin/python3

import json
import sys
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", action='store_false', help="Not sort by CIDEr")
parser.add_argument("-n", action='store_false', help="Not skip nan")
parser.add_argument("dirs", metavar='dir' ,nargs='+', help="model dirs")
args = parser.parse_args()

val_name = ["CIDEr","Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L"]
table = []
for start_filename in args.dirs:
    try:
        startx = 0
        filename = os.path.join(start_filename, 'model_.json')
        if not os.path.exists(filename):
            filename = os.path.join(start_filename, 'model_id.json')
        dirs = [filename]
        rrr = re.compile(r'\.t7$')
        if not(len(sys.argv) > 2 and sys.argv[2] == '-n'):
            while filename:
                data = json.load(open(filename))
                filename = rrr.sub('.json',data['opt']['start_from'])
                if filename and filename not in dirs and os.path.exists(filename):
                    dirs.append(filename)
                else:
                    break
            dirs.reverse()

        finetune_start = 0
        for filename in dirs:
            data = json.load(open(filename))
            if finetune_start == 0 and data['opt']['finetune_cnn_after'] != -1:
                finetune_start = data['opt']['finetune_cnn_after']
            val_max = {}
            max_cider = 0
            val_data = data["val_lang_stats_history"]
            val_data = sorted(val_data.items(),key=lambda t:int(t[0]))

            for name in val_name:
                max_result = 0
                for k,v in val_data:
                    if name == 'CIDEr' and float(v[name]) > max_result:
                        max_cider = startx + int(k)
                    max_result = max(max_result, float(v[name]))
                val_max[name] = max_result
            startx = startx + int(val_data[-1][0])
    except FileNotFoundError:
        if args.n:
            continue
        val_max = {i:float('NaN') for i in val_name}
        finetune_start = float('NaN')
        max_cider = float('NaN')
    table.append((start_filename, val_max, finetune_start, max_cider))

# output
max_name_len = max([len(i) for i in args.dirs])
print(" "*(max_name_len), end="")
for name in val_name:
    print("\t"+name, end="")
print("\t  pos(k)")
if args.s:
    table = sorted(table, key=lambda x:x[1]['CIDEr'], reverse=True)
for i in table:
    print(("%"+str(max_name_len)+"s")%os.path.basename(i[0]), end="")
    for name in val_name:
        print("\t%.3f"%i[1][name], end="")
    print("\t%4.0f+%3.0f"%((i[3]-i[2])/1000, i[2]/1000))
