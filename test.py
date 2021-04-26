from sklearn.ensemble import RandomForestClassifier
import math
import numpy as np


class dns_info:
    def __init__(self, _name, _is_dga=''):
        self.name = _name
        if _is_dga == 'dga':
            self.is_dga = 1
        else:
            self.is_dga = 0
        self.length = len(_name)
        self.digit_cnt = 0
        self.letter_cnt = 0
        self.entropy = 0
        self.__statistic__()

    def __statistic__(self):
        char_set = {x: self.name.count(x) for x in self.name}
        char_count = sum(char_set.values())
        for k, v in char_set.items():
            if k.isdigit():
                self.digit_cnt += v  # numbers in domain name
            if k.isalpha():
                self.letter_cnt += v  # letters in domain name
            prob = v / char_count
            self.entropy -= prob * math.log2(prob)  # entropy = sum[-p*log(p)]

    def return_value(self):
        return [self.name, self.length, self.digit_cnt, self.letter_cnt, self.entropy]


def read_and_preprocess_training_data(path):
    data = []
    tags = []
    with open(path, 'r') as f:
        for line in f.readlines():
            name, tag = line.strip().split(',')
            dns_obj = dns_info(name, tag)
            data.append(dns_obj.return_value()[1:])  # get the statistic results of the dns name
            tags.append(dns_obj.is_dga)
    return np.array(data), np.array(tags)


def read_and_preprocess_test_data(path):
    raw_data = []
    orig_domain = []
    with open(path, 'r') as f:
        for line in f.readlines():
            orig_domain.append(line.strip())
            dns_obj = dns_info(line)
            raw_data.append(dns_obj.return_value()[1:])  # get the statistic results of the dns name
    return np.array(raw_data), orig_domain


if __name__ == '__main__':
    data, tags = read_and_preprocess_training_data('train.txt')
    clf = RandomForestClassifier(random_state=0)
    clf.fit(data, tags)
    raw_data, orig_domain = read_and_preprocess_test_data('test.txt')
    results = clf.predict(raw_data)
    with open("result.txt", 'w') as f_w:
        for i in range(len(orig_domain)):
            if results[i] == 0:
                f_w.write("{},notdga\n".format(orig_domain[i]))
            else:
                f_w.write("{},dga\n".format(orig_domain[i]))
