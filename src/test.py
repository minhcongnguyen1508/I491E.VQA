import csv
import argparse

def read_csv(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    results = {}
    for row in csvreader:
        results.update({row[0]: row[1]}) 
    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='VQA',
                    epilog='Text at the bottom of help')
    parser.add_argument('--input', required=True)      # option that takes a value
    parser.add_argument('--test', required=False, default="BLIP")
    args = parser.parse_args()

    input_f = read_csv(args.input)
    test_f = read_csv(args.test)

    count = 0
    for k in test_f:
        if test_f[k].strip() == input_f[k].strip():
            count += 1

    print("Result: ", count/len(input_f))