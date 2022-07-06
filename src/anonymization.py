import os
import re
from collections import defaultdict
import json
import shutil
from datetime import datetime

from tqdm import tqdm
import openpyxl

from form import Form

class Anonymizer:

    def __init__(self, forms):
        self.forms = forms

        # infer the school_id for each teacher (2 sources)
        self.teacher_school_id = defaultdict(int)
        for form in forms:
            self.teacher_school_id[form.lecturer] = Form.school_id[form.institution]
            if form.auditor not in self.teacher_school_id and 'auditor_institution' in form.data:
                self.teacher_school_id[form.auditor] = Form.school_id[form.auditor_institution]

        # teacher <-> id
        self.teachers = set([form.lecturer for form in forms] + [form.auditor for form in forms])
        self.school_id_teachers = defaultdict(list)
        self.school_id_teacher_id = defaultdict(dict)

        self.school_id = Form.school_id
        self.id_school = Form.id_school
        self.teacher_strid = {}
        self.strid_teacher = {}
        for teacher in self.teachers:
            school_id = self.teacher_school_id[teacher]
            teacher_id = len(self.school_id_teachers[school_id])
            self.school_id_teachers[school_id].append(teacher)
            self.school_id_teacher_id[school_id][teacher] = teacher_id
            strid = f"{school_id:02d}{teacher_id:03d}"
            self.teacher_strid[teacher] = strid
            self.strid_teacher[strid] = teacher

    # Write the school - id, and teacher - id relationship
    def write_info(self, out_path):
        print(datetime.now(), f"saving anonymization info to {out_path}")
        with open(out_path + "teacher_ids.txt", 'w') as f:
            for strid, teacher in sorted(self.strid_teacher.items()):
                f.write(f"{strid}\t{teacher}\n")
        with open(out_path + "school_ids.txt", 'w') as f:
            for sid, school in sorted(self.id_school.items()):
                f.write(f"{sid:02d}\t{school}\n")

    # modify xlsx and mask all the private information
    def anonymize_xlsx(self, in_file, out_path):
        #wb = openpyxl.Workbook()
        date, filename_lecturer, filename_auditor = Form.get_filename_info(in_file)
        form = Form.from_excel(in_file)
        wb = openpyxl.load_workbook(in_file)
        ws = wb.worksheets[0]
        lecturer_id = self.teacher_strid[form.lecturer]
        auditor_id =  self.teacher_strid[form.auditor]

        # Iterate through the form, replace sensitive info in each cell 
        last = ''
        for row in ws.rows:
            for cell in row:
                if isinstance(cell.value, str):
                    if last.startswith('授课人') or last.startswith('听课人') or cell.value.startswith('授课人') or cell.value.startswith('听课人'):
                        cell.value = Form.clean_name_field(cell.value)
                    cell.value = cell.value.replace(form.lecturer, lecturer_id)
                    cell.value = cell.value.replace(form.lecturer[0]+"老师", lecturer_id+"老师")
                    cell.value = cell.value.replace(form.auditor, auditor_id)
                    cell.value = Form.school_pattern.sub(lambda m:  f"{Form.school_id[m.group()]:02d}", cell.value)
                    if 'auditor_remarks' in form.data and (cell.value.startswith("其他说明") or cell.value.startswith("综合观察") or cell.value.startswith("综合评价")):
                        cell.value += "（" + form.auditor_remarks + "）"
                    if last.startswith('授课人') or last.startswith('听课人'):
                        print(cell.value)
                    last = cell.value

        out_file = out_path + f"{form.date}-{lecturer_id}-{auditor_id}.xlsx"
        wb.save(out_file)
        return out_file
        
        
def main():
    original_path = 'original/'
    files = sorted(os.listdir(original_path))
    #files = ['20220419-王雷-黄建军.xlsx','20220413-王永才-虞强源.xlsx']
    forms = []
    for filename in tqdm(files):
        file_path = os.path.join(original_path, filename)
        print(file_path)
        form = Form.from_excel(file_path)
        forms.append(form)
    
    # for each form, mask info inside the xlsx, and change name
    anonymizer = Anonymizer(forms)
    anonymized_path = 'anonymized/'
    anonymized_forms = []
    if os.path.exists(anonymized_path):
        shutil.rmtree(anonymized_path)
    os.makedirs(anonymized_path, exist_ok=True) 
    for filename in tqdm(files):
    #for filename in files:
        in_file = os.path.join(original_path, filename)
        out_file = anonymizer.anonymize_xlsx(in_file, anonymized_path)
        print(in_file, '->', out_file)
        anonymized_forms.append(Form.from_excel(out_file))

    # write school-id and teacher-id relationship
    anonymiation_info_path = 'anonymization_info/'
    os.makedirs(anonymiation_info_path, exist_ok=True) 
    anonymizer.write_info(anonymiation_info_path)

    # write the json-line file
    anonymized_json_path = 'anonymized.json'
    print(datetime.now(), f"writing to {anonymized_json_path}")
    with open(anonymized_json_path, 'w') as f:
        for form in anonymized_forms:
            f.write(form.to_json_str())
            f.write('\n')
    
    
if __name__ == '__main__':
    main()
