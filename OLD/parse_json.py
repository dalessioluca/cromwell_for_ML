import json
import sys

new_dict = {}
with open(sys.argv[1], 'r') as f:
    my_dict = json.load(f)
    for k,v in my_dict.items():
        if k.startswith(sys.argv[2]):
           new_dict[k]=v
           #print(k,v)

print(json.dumps(new_dict))

with open("./params.json", 'w') as f:
        return json.dump(my_dict, f)
