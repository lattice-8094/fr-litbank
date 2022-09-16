import sys
import re

for fn in sys.argv[1:]:
    with open(fn, 'r+') as f:
        s = f.read()
    
    def invert(matchobj):
        if 'Arg1:' in matchobj.group() and 'Arg2:' in matchobj.group(0):
            old = matchobj.group().replace('\n','').split('\t')
            arg0 = old[1].split(' ')[0]
            arg1 = old[1].split(' ')[1].replace('Arg1:','')
            arg2 = old[1].split(' ')[2].replace('Arg2:','')
        return f'{old[0]}\t{arg0} Arg1:{arg2} Arg2:{arg1}\n'

    new_s = re.sub("R[0-9]{1,5}\t[^\n]*\n",invert,s)

    with open(fn.replace("coref", "coref_inverted"),'w+') as f:
        f.write(new_s)
