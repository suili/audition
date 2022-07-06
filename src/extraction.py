import os
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter
import argparse

# data structure
from form import Form

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--xlsx_path', type=str, default='original/', help='(input) folder of the excels (e.g. anonymized/)')
    parser.add_argument('--json_path', type=str, default='all.json', help='(output) json-lines file')
    parser.add_argument('--stats_path', type=str, default='stats/', help='(output) folder of the statistics of each field')
    parser.add_argument('--online_path', type=str, default='online/', help='(output) folder of the aggregated online/offline records')
    args = parser.parse_args()

    # STEP 1: parse the forms into "Form" objects, and store in all.json
    #files = ['20220301-邓俊辉-张铭.xlsx']
    files = sorted(os.listdir(args.xlsx_path))
    forms = []
    for filename in tqdm(files):
        file_path = os.path.join(args.xlsx_path, filename)
        print(file_path)
        form = Form.from_excel(file_path)
        forms.append(form)
    print(datetime.now(), f"writing to {args.json_path}")
    with open(args.json_path, 'w') as f:
        for form in forms:
            f.write(form.to_json_str())
            f.write('\n')
    
    # STEP 2: statistics of all the fields in the forms, to get intuition and check for mistakes
    os.makedirs(args.stats_path, exist_ok=True) 
    value_stat = defaultdict(list)
    print(datetime.now(), f"doing attributes statistics")
    for form in tqdm(forms):
        value_stat['date'].append(form.date)
        for key in Form.attr_dict.values():
            value_stat[key].append(form.data[key])
        value_stat['comment'].append(form.comment)
        for attr in Form.sub_attrs:
            for s in form.sub_data[attr]:
                if s != 'None':
                    value_stat[f"sub_{attr}"].append(s)
    print(datetime.now(), f"writing to {args.stats_path}")
    for key in value_stat.keys():
        with open(args.stats_path + f"{key}.txt", 'w') as f:
            counter = Counter(value_stat[key])
            items = sorted(counter.items(), key=lambda p:(-p[1], p[0]))
            for s, count in items:
                f.write(f"{count}\t{s}\n")
    
    # STEP 3: how many of these auditions are online/offline?
    os.makedirs(args.online_path, exist_ok=True) 
    print(datetime.now(), f"stating and writing online data into {args.online_path}")
    online = 0
    offline = 0
    both = 0
    unknown = 0
    with open(args.online_path + "online.txt", 'w') as f_online, open(args.online_path  + "offline.txt", 'w') as f_offline, open(args.online_path  + "both.txt", 'w') as f_both, open(args.online_path  + "unknown.txt", 'w') as f_unknown: 
        for form in tqdm(forms):
            if form.online and not form.offline:
                f_online.write(form.to_json_str())
                f_online.write('\n')
                online += 1
            if not form.online and form.offline:
                f_offline.write(form.to_json_str())
                f_offline.write('\n')
                offline += 1
            if form.online and form.offline:
                f_both.write(form.to_json_str())
                f_both.write('\n')
                both += 1
            if not form.online and not form.offline:
                f_unknown.write(form.to_json_str())
                f_unknown.write('\n')
                unknown += 1
    print(f"online: {online}, offline: {offline}, both: {both}, unknown: {unknown}, total: {len(forms)}")
    


if __name__ == '__main__':
    main()
