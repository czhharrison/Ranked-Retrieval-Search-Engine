import os, sys, re, json, itertools
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class SearchEngine:
    ALPHA = 1.0
    BETA  = 1.0
    GAMMA = 0.1          # ordered pairs weight

    def __init__(self, index_folder: str):
        self.index_folder = index_folder                # inverted index
        self.lemmatizer = WordNetLemmatizer()
        self.inverted_index: dict[str, list[list]] = {} # data structure of inverted data
        self.load_index()                              # load index file

    # load inverted index
    def load_index(self) -> None:
        with open(os.path.join(self.index_folder, "inverted_index.json")) as f:
            self.inverted_index = json.load(f)          # load from json file

    # generate all root for a word
    def find_root(self, word: str) -> set[str]:
        root = {
            self.lemmatizer.lemmatize(word),                        # default lemmatization
            self.lemmatizer.lemmatize(word, pos=wordnet.NOUN),      # lemmatization as noun
            self.lemmatizer.lemmatize(word, pos=wordnet.VERB),      # lemmatization as verb
        }
        return root

    # normalize token
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
            return tokens
        
        # normal English word
        if token.isalnum():
            return list(self.find_root(token))
        
        return []   # drop other symbols

    # preprocess the sentence
    def preprocess_sentence(self, sentence: str) -> list[str]:
        sentence = re.sub(r"'s|s'", "", sentence)           # remove possessives
        sentence = re.sub(r"(\d{1,3})(,\d{3})+", lambda m: m.group().replace(",", ""), sentence)     # remove the comma in thousands number
        
        terms = []
        for i in word_tokenize(sentence):
            terms.extend(self.normalize_token(i))       # expand multiple word forms
        
        return list(dict.fromkeys(terms))           # remove duplicates, keep order

    # search the best matching combination for shortest distance
    def shortest_distance(self, search_term: list[str], position: dict[str, list[int]]):
        match_word  = [t for t in search_term if t in position]     # query terms actual hit
        match_list  = [position[t] for t in match_word]             # all matching positions for each term

        if not match_word:
            return float("inf"), 0, {}      # if all miss

        best_distance  = float("inf")           # best total distance
        max_number  = -1                        # max number of ordered word pairs
        best_position = None                    # best position combination

        # for all combination
        for c in itertools.product(*match_list):
            position_dic = dict(zip(match_word, c))

            c = sorted(c)           # sort token position in ascending order
            sum_int = sum(c[i + 1] - c[i] - 1
                          for i in range(len(c) - 1))       # caculate sum of token interval 

            # count ordered pairs based on original search_term sequence
            pairs_num = 0
            for i in range(len(search_term) - 1):
                term1, term2 = search_term[i], search_term[i + 1]
                if term1 in position_dic and term2 in position_dic:
                    if position_dic[term1] < position_dic[term2]:
                        pairs_num += 1

            # update best, shorter or more order
            if (sum_int < best_distance) or (
                sum_int == best_distance and pairs_num > max_number
            ):
                best_distance, max_number, best_position = sum_int, pairs_num, position_dic

        avg_distance = best_distance / max(len(match_word) - 1, 1)      # average token interval
        return avg_distance, max_number, best_position


    def search(self, origin_query: str):
        match_line = origin_query.startswith(">")          # show match line if >
        query = origin_query[1:].strip() if match_line else origin_query.strip()
        search_term = self.preprocess_sentence(query)      # preprocess query
        
        if not search_term:
            return []           # query word is empty


        match = defaultdict(lambda: defaultdict(list))       # file to the terms
        for t in search_term:
            if t in self.inverted_index:
                for id, pos, line in self.inverted_index[t]:
                    match[id][t].append((pos, line))

        temp_rank = []
        for id, pos in match.items():
            match_term = list(pos.keys())                      # match terms in current file
            coverage = len(match_term) / len(search_term)      # coverage

            if len(match_term) == 1:            # only one term matched
                cur_score, pairs_num, chosen = 0, 0, {
                    match_term[0]: pos[match_term[0]][0][0]     # get the first position
                }
            else:
                position = {t: [p for p, _ in pos[t]] for t in match_term}      # all position of each term
                avg_distance, pairs_num, chosen = self.shortest_distance(search_term, position)    # calculate min avg distance and ordered pairs
                cur_score = 1 / (1 + avg_distance)          # shorter distance, higher score
            
            # calculate final score
            score = (self.ALPHA * coverage +
                     self.BETA  * cur_score +
                     self.GAMMA * pairs_num)

            temp_rank.append((id, score, chosen))       # save file id, score, best match position

        # remove duplicate keep the highest score
        unique = {}
        for id, score, chosen in temp_rank:
            if (id not in unique) or (score > unique[id][0]):
                unique[id] = (score, chosen)            # update tot the higher score combination

        ranked = [(d, *info) for d, info in unique.items()]     # change stucture to list
        # Fix tie-breaking with floating point precision tolerance
        ranked.sort(key=lambda x: (-round(x[1], 10), int(x[0])))           # descending order, if same score, ascending order by id

        # output result
        printed = set()             # file id already output
        for id, score, chosen in ranked:
            assert id not in printed, f"dup doc {id}"       # remove duplicate
            printed.add(id)
            if match_line:          # if >
                print(f"> {id}")
                self.print_match(id, chosen)       # print matched line
            else:
                print(id)           # only output id

        return ranked               # return sorted result list

    # output the matching line
    def print_match(self, id: str, chosen: dict[str, int]):
        file_path = os.path.join(self.index_folder, "doc", id)      # set up file path
        if not os.path.isfile(file_path):           # file missing
            return
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()                   # read all lines

        line_id = set()                             # record the line id to print
        for term, pos in chosen.items():
            # Find all lines containing this term at the chosen position
            for n, p, i in self.inverted_index.get(term, []):   # for all record of the term in inverted index
                if n == id and p == pos:            # whether current position is chosen position
                    line_id.add(i)                  # add line id
                    # Don't break - there might be multiple lines with same term at same position

        for i in sorted(line_id):               # ascending order by line id
            if 0 <= i < len(lines):
                line = lines[i]
                if not line.endswith("\n"):     # wrap at each line end
                    line += "\n"
                print(line, end="")             # print content



def main() -> None:
    if len(sys.argv) != 2:          # check parameter number in terminal
        print("Usage: python3 search.py [folder-of-indexes]")
        sys.exit(1)

    idx = sys.argv[1]
    if not os.path.isdir(idx):          # check whether index folder exist
        print(f"Error: '{idx}' is not a directory", file=sys.stderr)
        sys.exit(1)

    engine = SearchEngine(idx)          # initial search engine
    for i in sys.stdin:                 # read input from normalize input
        input = i.rstrip("\n")          # remove wrap
        if input:
            engine.search(input)        # execute search 


if __name__ == "__main__":
    main()
