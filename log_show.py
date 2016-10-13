#!/usr/bin/python3

import json
import sys
import os
import re

val_name = ["CIDEr","Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L"]
print(" "*20, end="")
for name in val_name:
    print("\t"+name, end="")
print("\t  pos(k)")
for start_filename in sys.argv[1:]:
    try:
        startx = 0
        filename = os.path.join(start_filename, 'model_.json')
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
        val_max = {i:float('NaN') for i in val_name}
        finetune_start = float('NaN')
        max_cider = float('NaN')
    print("    %-16s"%start_filename, end="")
    for name in val_name:
        print("\t%.3f"%val_max[name], end="")
    print("\t%4.0f+%3.0f"%((max_cider-finetune_start)/1000, finetune_start/1000))
