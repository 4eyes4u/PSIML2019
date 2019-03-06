import os
import io
import sys

def detect_lang_from_path(path):
    for lang in langs:
        if lang in path:
            return lang

def extract_bigrams_from_string(data):
    invalid = ['\r', '\n']
    str_len = len(data)
    bigrams = []
    for i, j in zip(range(1, str_len), range(str_len - 1)):
        if data[i] not in invalid and data[j] not in invalid:
            bigrams.append(''.join([data[j], data[i]]))
    return bigrams

def extract_bigrams_from_file(file_path):
    # TODO: KEEP EYE ON ENCODING
    with io.open(file_path, 'r', encoding="utf-8") as f:
        data = f.read().lower()
        bigrams = extract_bigrams_from_string(data)
    
    return bigrams

def pick_best_sample(compress, k=5):
    '''
        Picking by #occurrence first then by lex. order
    '''
    sample = [(val, key) for key, val in compress.items()]
    sample = sorted(sample, key=lambda entry: (-entry[0], entry[1]))
    return sample[ : k]

def compress_data(raw):
    '''
        Putting into dictionary
    '''
    cnt = dict()
    for bi in raw:
        old_val = cnt.get(bi)
        cnt[bi] = 1 if old_val is None else old_val + 1

    return cnt

def print_bigrams():
    for lang in langs:
        sample = pick_best_sample(data_compress[lang])
        for entry in sample:
            print('{},{},{}'.format(lang, entry[1], entry[0]))

def calc_probs():
    '''
        Frequency ratio
    '''
    for lang in langs:
        total = sum([val for val in data_compress[lang].values()])
        for key, val in data_compress[lang].items():
            data_probs[lang].update({key: float(val) / float(total)})

def text_prob(bigrams, p_langs_given_text):
    '''
        P(text) = Product_{bi} Sum_{lang} #(bi[lang]) / ultimate_total 
    '''
    for bi in bigrams:
        n_bi = 0
        for lang in langs:
            if bi in data_compress[lang]:
                n_bi += data_compress[lang][bi]
        for lang in langs:
            if n_bi != 0:
                p_langs_given_text[lang] *= float(ultimate_total / n_bi)
            else:
                p_langs_given_text[lang] = 0.0

    return p_langs_given_text    

def text_given_lang_prob(bigrams, lang_prob):
    p = 1.0
    for bi in bigrams:
        p = p * (lang_prob[bi] if bi in lang_prob else 0.0)
    return p

def normalize(dict_prob):
    p_total = sum([p for p in dict_prob.values()])
    if p_total != 0:
        dict_prob = {key: p / p_total for key, p in dict_prob.items()}
    return dict_prob

if __name__ == "__main__":
        corpus_dir = input()
        langs = [lang for lang in os.listdir(corpus_dir)]
        n_langs = len(langs)
        default_prob = float(1 / n_langs)
        langs.sort()
        data_raw = {lang: [] for lang in langs}
        data_compress = {lang: dict() for lang in langs}
        data_probs = {lang: dict() for lang in langs}
        data_total = {lang: 0 for lang in langs}
        ultimate_total = 0.0

        for lang in langs:
            lang_dir = os.path.join(corpus_dir, lang)
            for root, dirs, files in os.walk(lang_dir):
                for f in files:
                    cur_path = os.path.join(root, f)
                    cur_bigrams = extract_bigrams_from_file(cur_path)
                    data_raw[lang] += cur_bigrams

        for lang in langs:
            data_compress[lang] = compress_data(data_raw[lang])
            data_total[lang] = sum([val for val in data_compress[lang].values()])
            ultimate_total += data_total[lang]
        
        # PRINTING FIRST TASK
        print_bigrams()

        calc_probs()
        sequence_path = input()
        with io.open(sequence_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                bigrams = extract_bigrams_from_string(line.lower())

                p_langs_given_text = {lang: text_given_lang_prob(bigrams, data_probs[lang]) for lang in langs}
                text_prob(bigrams, p_langs_given_text)
                p_langs_given_text = normalize(p_langs_given_text)
                for lang in langs:
                    print('{},{}'.format(lang, round(p_langs_given_text[lang], 18)))