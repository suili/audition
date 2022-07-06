import re
from collections import defaultdict
import json

import openpyxl

class Form:
    attr_dict = {
        "时间":"time",
        "授课学校":"institution",
        "授课人":"lecturer",
        "听课人":"auditor",
        "课程":"course",
        "章节（主题）":"topic",
        "应到学生数":"student_total",
        "实到学生数":"student_present"
    }
    attr_pattern = re.compile(f"({'|'.join(f'{k}' for k in attr_dict.keys())})(?:.*?)(?:：|:) *(.+?)None")

    online_pattern = re.compile('|'.join(["在线","线上","弹幕","直播","远程","会议","聊天室","聊天区","网上","录像","看不到","腾讯","无法观察","关麦","不能线下","摄像头","网络原因"]))
    offline_pattern = re.compile('|'.join(["低头","睡觉","抬头","板书","教室","作笔记","记笔记","认真","走神","聆听","看书","上台","前排","后排","左右看","点头", "黑板"]))

    # 教学环节 时长 教师教学 学生学习 亮点 建议 备注
    sub_attrs = ['step', 'length', 'teacher', 'student', 'highlight', 'suggestion', 'note']

    schools = ['北京大学','北航','北交大','北理工','北邮','大连理工','成电科大','东南大学', '复旦大学','国防科大','哈工大','湖南大学','华东师大','华南理工','华中科大','吉林大学', '南京大学','清华大学','山东大学','上交大','天津大学','同济大学','武汉大学','西电科大', '西交大','西工大','浙江大学','中科大','国科大','人大','中南大学','中山大学','重庆大学']

    alias = {"上海交通大学":"上交大", "国防科技大学":"国防科大", "中国科学院大学":"国科大", "西安电子科技大学":"西电科大", "北京交通大学":"北交大", "北京航空航天大学":"北航", "华中科技大学":"华中科大", "西安交通大学":"西交大", "哈尔滨工业大学":"哈工大", "北京理工大学":"北理工", "中国人民大学":"人大", "中国科学技术大学":"中科大", "北京航天航空大学":"北航", "北京邮电大学":"北邮", "大连理工大学":"大连理工", "清华":"清华大学", "电子科技大学":"成电科大", "中科院大学":"国科大", "人民大学":"人大", "华东师范大学":"华东师大", "华南理工大学":"华南理工", "西北工业大学":"西工大", "中南大学（新校区）":"中南大学", "中国科技大学":"中科大", "南京 大学":"南京大学", "国防科学技术大学":"国防科大", "西北工大":"西工大", "西电":"西电科大"}
    school_id = {a:b + 1 for a,b in zip(schools, range(len(schools)))} # must do this due to list comprehension implementation
    id_school = {b + 1:a for a,b in zip(schools, range(len(schools)))} # must do this due to list comprehension implementation

    for a in alias.keys():
        school_id[a] = school_id[alias[a]]
    school_pattern = re.compile('|'.join(sorted(school_id.keys(), key=lambda x: -len(x))))  # match longer names first

    # added last, do not go into pattern
    school_id[''] = 0
    id_school[0] = ''

    def __init__(self, data):
        self.data = data

    # MAIN METHOD: parse an excel file
    @classmethod
    def from_excel(cls, xlsx_path, debug=False):
        data = {}  # result data stored here
        
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.worksheets[0]
        rows = list(ws.rows)
        all_str = ' '.join(f"{c.value}" for row in rows for c in row)

        # === part 0: filename contains most accurate date ===
        data['date'], filename_lecturer, filename_auditor = cls.get_filename_info(xlsx_path)

        # === part 1: 8 basic attributes as in attr_dict
        meta_str = ' '.join(f"{c.value}" for row in rows[1:7] for c in row)
        missing = set(Form.attr_dict.keys())  
        for k, v in Form.attr_pattern.findall(meta_str):
            data[Form.attr_dict[k]] = v.strip()
            missing.remove(k)
        # bad case
        if missing: 
            print("WARNING: Missing info ", "\t".join(missing))
            print(meta_str)
        # if the auditor wrote additional information in "auditor" box, store them
        tmp = re.search(cls.school_pattern, data['auditor']) 
        if tmp is not None:
            data['auditor_institution'] = tmp.group(0)
        tmp = re.search(cls.online_pattern, data['auditor']) 
        if tmp is not None:
            data['auditor_remarks'] = tmp.group(0)
        # lecturer/auditor fields needs to be cleaned (very messy)
        data['lecturer'] = cls.clean_name_field(data['lecturer'], filename_lecturer)
        data['auditor'] = cls.clean_name_field(data['auditor'], filename_auditor)
        if data['auditor'] == "张莉 张莉":
            data['auditor'] =  "张莉"
        
        # === part 2: parse teaching steps and final comment
        data['sub_data'] = defaultdict(list)
        begin_idx, end_idx, comment_idx = -1, -1, -1
        for i in range(len(rows)):
            row = rows[i]
            row_str = ' '.join(f"{c.value}" for c in row)
            if begin_idx == -1 and row_str.startswith("教学环节"):
                begin_idx = i + 1
                continue
            if begin_idx != -1 and end_idx == -1 and all(c.value is None for c in row):
                end_idx = i
            if row_str.startswith("其他说明") or row_str.startswith("综合观察") or row_str.startswith("综合评价"):
                comment_idx = i
                if end_idx == -1:
                    end_idx = i

        # bad case warnings, could normal
        if debug and (begin_idx == -1 or end_idx == -1 or begin_idx == end_idx):
            print(f"WARNING: No teaching steps! {begin_idx}:{end_idx}")
            print(all_str)
        if debug and (comment_idx == -1 and rows[-1][0] is None):  
            print("WARNING: No comment!")
            print(all_str)
        
        # store teaching steps data
        for i in range(begin_idx, end_idx):
            row = rows[i]
            for j in range(7):
                str_value = '' if row[j].value is None else str(row[j].value)
                data['sub_data'][Form.sub_attrs[j]].append(str_value.strip().replace("\n", " "))

        # store comment data
        comment = rows[comment_idx][0].value
        if comment is None:
            comment = ""
        if comment.startswith("综合") or comment.startswith("其他"):
            idx = comment.find("：")
            if idx >= 0 and idx <= 40:
                comment = comment[idx + 1:]
        data['comment'] = comment.strip().replace("\n", " ")

        # === part 3: derived features ===
        
        data['online'] = Form.online_pattern.search(all_str) is not None
        data['offline'] = Form.offline_pattern.search(all_str) is not None
        
        data['length'] = len(re.sub(r'None|\s', '', all_str))
            
        return cls(data)


    # parse information from filenames, used as backup
    @classmethod
    def get_filename_info(cls, filename):
        # 20220301-邓俊辉-张静.xlsx
        filename = filename.replace('_','-')
        filename = re.sub(r'-+', '-', filename)
        begin = filename.find('/')
        end = filename.find('-')
        date = filename[begin + 1: end].strip()

        begin2 = filename.rfind('-')
        end2 = filename.find('.')
        filename_lecturer = cls.clean_name_field(filename[end + 1: begin2])
        filename_auditor =  cls.clean_name_field(filename[begin2 + 1: end2])
        return date, filename_lecturer, filename_auditor

    # clean the auditor/lecturer box
    @classmethod
    def clean_name_field(cls, name, alternative=''):
        name = name.replace("张莉（南京大学）", "张莉[南大]")
        name = re.sub("[\(（].*?[\)）]", "", name)
        name = name.split('-')[0]
        name = name.replace("老师","").replace("教授","")
        name = name.replace("现场：","").replace("网上：","")
        name = Form.school_pattern.sub("", name)
        name = " ".join(re.split(" +|、|，", name))
        name = name.strip()
        return name if len(name)>0 and name != "None"  else alternative

    @classmethod
    def from_json_str(cls, json_str):
        data = json.loads(json_str)
        return cls(data)

    def to_json_str(self):
        return json.dumps(self.data, ensure_ascii=False)

    def __getattr__(self, key):
        return self.data[key]

    # concatenate all "highlights" in each teaching steps
    def highlights(self):
        return " ".join(self.sub_data['highlight']) if 'highlight' in self.sub_data else ''

    # concatenate all "suggestions" in each teaching steps
    def suggestions(self):
        return " ".join(self.sub_data['suggestion']) if 'suggestion' in self.sub_data else ''

    def comments(self):
        return self.comment

    # all "highlights + suggestions" + comment
    def all_remarks(self):
        return re.sub("\s+" , " ", " ".join([self.highlights(), self.suggestions(), self.comments()]))
        
if __name__ == '__main__':
    pass
