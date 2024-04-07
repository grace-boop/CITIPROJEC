import pdfplumber
import nltk
import fitz  # PyMuPDF
import os
import re
import numpy as np
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

#Intialization
nltk.download('punkt')
nltk.download('wordnet')
word2vec_model = api.load("word2vec-google-news-300")
#wordlist:關鍵字list，可以改!!!!!
wordlist = ["block-chain-related","cloud technology","data technology",
"internet technology","blockchain","cloud computing","big data",
"mobile","alliance chain","cloud architecture","data layer","internet",
"test chain","cloud service","dataset","network","interconnected chain",
"cloud finance","data flow","online"]

#stoplist:if出現這些字則句子不算數，可以改!!!!
stoplist = ["not","no"]
'''
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    lemmatized_sentence = " ".join(lemmatized_tokens)
    return lemmatized_sentence

def caculate_vector(sentence):
    tokens = sentence.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if not word_vectors:
        return None
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def sentence_similarity(sentence_vector, word_vector):
    similarities = [cosine_similarity(word, word_vector) for word in sentence_vector]
    return np.mean(similarities)
'''

def citiproject(file):
    with fitz.open('./data/'+file) as pdf_document:
        #word_frequency = [0 for _ in range(len(wordlist))] 
        my_dict = {}
        #textLIst: one index one page 
        textlist = [] 
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            textlist.append(text)

        word_matrix = [[0 for _ in range(len(textlist))] for _ in range(len(wordlist))]
        # len(wordlist) rows, len(textlist) columns??
        for i in range(len(wordlist)):
            ##appear in which pages
            page_number = []
            ##appear in which sentences
            sen_list = []
            word_vector = word2vec_model[wordlist[i]]
            print(wordlist[i])
            print(word_vector)
            for j in range(len(textlist)):
                #appear in which sentences, an index is in the same page_number
                seq=[]
                sentences = nltk.sent_tokenize(textlist[j])#沒有切詞，不知道會不會有差
                for sentence in sentences:
                    #word_in_sent = sentence.lower().count(wordlist[i])
                    pattern = r"\b"+re.escape(wordlist[i])+r"\b"
                    lemmatized_sentence = lemmatize_sentence(sentence)
                    word_in_sent = len(re.findall(pattern, lemmatized_sentence, re.IGNORECASE))
                    if word_in_sent>0:
                        #determine whether the sentence has stopwprds or not, stop_flag = 1 if its has stopword
                        stop_flag = 0
                        for stop in stoplist:
                            temp_sent = lemmatized_sentence.split()
                            if stop.lower() in [w.lower() for w in temp_sent]:
                                stop_flag = 1
                        #determine whether the sentence is closed to word or not
                        sentence_vector = caculate_vector(lemmatized_sentence)
                        similarity = sentence_similarity(sentence_vector, word_vector)
                        print(lemmatized_sentence)
                        print(similarity)
                        if stop_flag == 1:
                            continue
                        else:
                            word_matrix[i][j]+=word_in_sent
                            seq.append(sentence)###有換行符號 在看看要怎樣
                        
                #the sentences append in 
                if len(seq)>0:
                    sen_list.append(seq)
                #the Ith word is appear in the Jth page
                if word_matrix[i][j] > 0:
                    page_number.append(j) 
            temp_count = 0
            for j in range(len(textlist)):
                temp_count+=word_matrix[i][j]
            new_row = {"frequence":temp_count,"page_number":page_number,"sentences":sen_list}
            my_dict.update({wordlist[i]:new_row})

        #print(my_dict)

        '''
        for i in range(len(wordlist)):
            print(wordlist[i])
            print(my_dict[wordlist[i]])
            print("\n")
        '''

        ##存到excel
        wb = Workbook()
        ws = wb.active
        ws.append(["word", "frequence", "page_number", "sentences"])
        row_idx = 2
        for word, word_data in my_dict.items():

            ws.cell(row=row_idx, column=1, value=word)
            #if not word_data["sentences"]:
            #    continue  # 如果为空列表，则跳过当前行的写入

            for column_idx, (key, value) in enumerate(word_data.items(), start=2):
                # 如果值是列表，则将列表中的每个元素写入到相应的列中
                if isinstance(value, list):
                    if key == "page_number":
                        for idx, page_number in enumerate(value, start=1):
                            ws.cell(row=row_idx + idx -1, column=column_idx, value=page_number)
                    else:
                        for sublist_idx, sublist in enumerate(value, start=1):
                            for subvalue_idx, subvalue in enumerate(sublist, start=1):
                                cell = ws.cell(row=row_idx + sublist_idx - 1, column=column_idx + subvalue_idx - 1, value=subvalue)

                                column_letter = get_column_letter(cell.column)
                                ws.column_dimensions[column_letter].width = max(ws.column_dimensions[column_letter].width, len(subvalue)+2)
                else:
                    ws.cell(row=row_idx, column=column_idx, value=value)
            row_idx +=max(1, len(word_data["sentences"]))
        # 保存 Excel 文件
        wb.save("./result/"+file+".xlsx")
        
folder_path = "./data/"
files = os.listdir(folder_path)
for file in files:
    citiproject(file)
    print(file)

#pdf plumber

'''
with pdfplumber.open('./data/Citi-2023-Annual-Report.pdf') as pdf:
    textlist = []

    for page in pdf.pages:
        text = page.extract_text()
        textlist.append(text)
        ##print("'"+text+"'")
        ##print("\n")
        ##print(type(text))
        ##print(text[-1]
'''
