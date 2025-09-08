import os
import sys
import re
import json
import shutil
from collections import defaultdict

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# read file and build inverted index
class FileProcess:

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.inverted_index: dict[str, list[list]] = defaultdict(list)    # inverted index
        self.unique_words: set[str] = set()        # all unique words appeared
        self.num_file = 0              # count the number of files

    # generate all root for a word
    def find_root(self, word: str) -> set[str]:
        root = {
            self.lemmatizer.lemmatize(word),                        # default lemmatization
            self.lemmatizer.lemmatize(word, pos=wordnet.NOUN),      # lemmatization as noun
            self.lemmatizer.lemmatize(word, pos=wordnet.VERB),      # lemmatization as verb
        }
        return root

    # normalize tokens into a set to use in index
    def normalize_token(self, token: str) -> list[str]:
        token = token.lower()           # lower case
        token = token.replace("'s", "").replace("s'", "")   # remove possessives

        # process abbreviation
        if re.fullmatch(r"[a-z]\.[a-z]\.", token):
            return [token.replace(".", "")]

        # keep numebers
        if token.isdigit():
            return [token]

        # process '-'
        if "-" in token:
            splited_token = token.split("-")
            if len(splited_token[0]) < 3:          # short prefix keep original word
                return list(self.find_root(token))
            
            # separate each part by '-'
            tokens = []
            for i in splited_token:
                if i.isalnum():
                    tokens.extend(self.find_root(i))
            return list(tokens)

        # normal English word
        if token.isalnum():
            return list(self.find_root(token))

        return []  # drop other symbols

    # preprocess the sentences
    def preprocess_sentence(self, sentence: str) -> str:
        sentence = re.sub(r"'s|s'", "", sentence)           # remove possessives
        sentence = re.sub(r"(\d{1,3})(,\d{3})+", lambda m: m.group().replace(",", ""), sentence)     # remove the comma in thousands number
        return sentence

    # tokenize the sentence and return result
    def tokenize_sentence(self, sentence: str):
        for i in word_tokenize(sentence):             # tokenize
            for t in self.normalize_token(i):         # per word
                yield t               # return normalized token 

    # create inverted index for single file
    def file_index(self, id: str, file_path: str) -> None:
        position = 0                # initial global token position
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for n, line in enumerate(f.read().splitlines()):      # read every line of file
                for cur in sent_tokenize(line):                        # split to sentence
                    cur = self.preprocess_sentence(cur)               # preprocess sentence
                    for t in self.tokenize_sentence(cur):              # tokenize sentence
                        self.inverted_index[t].append([id, position, n])      # inverted index record
                        self.unique_words.add(t)                        # add token into the unique word set
                        position += 1                                   # update token position

    # save inverted inddex
    def save_index(self, index_folder: str) -> None:
        with open(os.path.join(index_folder, "inverted_index.json"), "w") as f:
            json.dump(self.inverted_index, f)

    # build index and process file
    def build_index(self, file_folder: str, index_folder: str) -> None:
        os.makedirs(index_folder, exist_ok=True)            # create index
        temp_file = os.path.join(index_folder, "doc")       # subfolder to save all files
        os.makedirs(temp_file, exist_ok=True)               # create subfolder

        for name in sorted(os.listdir(file_folder)):        # for every file in folder
            ori_path = os.path.join(file_folder, name)      # original path of file
            if not os.path.isfile(ori_path):                # ignore no file entries
                continue
            target_folder = os.path.join(temp_file, name)       # target path after copy
            shutil.copyfile(ori_path, target_folder)            # copy file to index
            self.file_index(name, target_folder)                # create inverted index for file
            self.num_file += 1                                  # update number of files  

        self.save_index(index_folder)               # save inverted index

        # calculate output
        total_tokens = sum(len(v) for v in self.inverted_index.values())
        total_terms = len(self.inverted_index)
        print(f"Total number of documents: {self.num_file}")
        print(f"Total number of tokens: {total_tokens}")
        print(f"Total number of terms: {total_terms}")


def main() -> None:
    if len(sys.argv) != 3:          # check parameter number in terminal
        print("Usage: python3 index.py [folder-of-documents] [folder-of-indexes]")
        sys.exit(1)

    input_path, index_path = sys.argv[1], sys.argv[2]       # get input path
    if not os.path.isdir(input_path):                       # check whether folder
        print(f"Error: '{input_path}' is not a directory", file=sys.stderr)
        sys.exit(1)

    FileProcess().build_index(input_path, index_path)       # call class function


if __name__ == "__main__":
    main()