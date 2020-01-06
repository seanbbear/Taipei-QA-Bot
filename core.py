import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

def makeDataset(input_ids, input_masks, input_segment_ids, answer_lables):
    all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
    all_input_masks = torch.tensor([input_mask for input_mask in input_masks], dtype=torch.long)
    all_input_segment_ids = torch.tensor([input_segment_id for input_segment_id in input_segment_ids], dtype=torch.long)
    all_answer_lables = torch.tensor([answer_lable for answer_lable in answer_lables], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_masks, all_input_segment_ids, all_answer_lables)

    # 切分Dataset成train, test https://kknews.cc/zh-tw/code/b8833zo.html
    train_num = int(0.8 * len(dataset))
    valid_num = len(dataset)-train_num    
    train_dataset, valid_dataset = random_split(dataset, [train_num, valid_num])

    return train_dataset,valid_dataset



class AnsDic(object):
    def __init__(self, answers):
        self.answers = answers #全部答案(含重複)
        self.answers_norepeat = sorted(list(set(answers))) # 不重複
        self.answers_types = len(self.answers_norepeat) # 總共多少類
        self.ans_list = [] # 用於查找id或是text的list
        self._make_dic() # 製作字典
    
    def _make_dic(self):
        for index_a,a in enumerate(self.answers_norepeat):
            if a != None:
                self.ans_list.append((index_a,a))

    def to_id(self,text):
        for ans_id,ans_text in self.ans_list:
            if text == ans_text:
                return ans_id

    def to_text(self,id):
        for ans_id,ans_text in self.ans_list:
            if id == ans_id:
                return ans_text

    @property
    def types(self):
        return self.answers_types
    
    @property
    def data(self):
        return self.answers

    def __len__(self):
        return len(self.answers)

class QuestionDic(AnsDic):
    def __init__(self,questions):
        super().__init__(answers = questions)

