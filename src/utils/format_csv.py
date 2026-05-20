import csv
import numpy as np



def edit_csv_line_by_line(input_path, output_path, edit_function):
    with open(input_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)


        for row in reader:       
            new_row = []
            for array in row :
                new_array = edit_function(array)
                new_row.append(new_array)
            writer.writerow(row)

        
def parse_vector(cell):
    # Nettoyage
    cell = cell.replace('[', '').replace(']', '').replace(',','').strip()
    
    # Split intelligent (ignore les espaces multiples)
    values = cell.split()
    
    # Conversion en float
    return [float(v) for v in values]


"""
for d in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']: 

    edit_csv_line_by_line("Dataset_csv_2/Train/" + d + ".csv", "DATASET_CSV/Train/" + d + ".csv", parse_vector)
    edit_csv_line_by_line("Dataset_csv_2/Test/" + d + ".csv", "DATASET_CSV/Test/" + d + ".csv", parse_vector)
    edit_csv_line_by_line("Dataset_csv_2/Validation/" + d + ".csv", "DATASET_CSV/Validation/" + d + ".csv", parse_vector)
"""





