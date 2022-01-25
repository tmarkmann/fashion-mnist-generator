import csv

DATA_FILE = '/Users/tmarkmann/Downloads/test.csv'

count = {
    'real': 0,
    'tfgan': 0,
    'cvae': 0,
    'lsgm': 0,
    'stylegan2': 0,
}
correct = {
    'real': 0,
    'tfgan': 0,
    'cvae': 0,
    'lsgm': 0,
    'stylegan2': 0,
}

with open(DATA_FILE, mode='r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for line_count, row in enumerate(csv_reader):        
        image_type = row['type']
        resp = row['key_resp_3.keys']
        if image_type == '' or resp == '':
            continue
        
        count[image_type] += 1
        if image_type == 'real' and resp == 'left':
            correct[image_type] += 1
        elif image_type != 'real' and resp == 'right':
            correct[image_type] += 1

print(f'count: {count}')
print(f'correct: {correct}')