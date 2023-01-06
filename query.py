import sys
import re
import file_io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List

'''
Querier class which handles the "query" function of Search. The user inputs
filepaths for title, doc, and words files, along with an optional argument
--pagerank to include pagerank in calculations. Runtime should be much faster
than index.py. Includes a main method, which creates instances of Querier.
'''
class Querier:

    '''
    Constructor for the querier class. Sets up the dictionaries of ids->titles,
    ids->pageranks, and words->ids->term relevance. Also initializes the 
    pagerank variable which is a boolean. Lastly, it starts the REPL by calling
    self.query().
    '''
    def __init__(self, pagerank: bool, titles_filepath: str, docs_filepath: \
        str,
        words_filepath: str):
        self.titles_dict = {}
        self.docs_dict = {}
        self.words_dict = {}

        self.pagerank = pagerank

        #populate dicts
        file_io.read_title_file(titles_filepath, self.titles_dict)
        file_io.read_docs_file(docs_filepath, self.docs_dict)
        file_io.read_words_file(words_filepath, self.words_dict)

        #start the REPL
        self.query()

    '''
    The query method represents our REPL. The user is prompted for input,
    then the input query is processed and answered. The user is prompted
    again to enter another query or enter :quit to quit. The while loop
    continues until :quit is entered. 
    '''
    def query(self):
        print('Input a query!')
        user_input = input()
        while user_input!=':quit':
            print()
            print('Your results:')
            self.answer_query(self.process_input(user_input))
            print('Input another query, or :quit to quit.')
            user_input = input()

    '''
    This determines and prints the results for the user's query. For each
    page in the wiki, the total score is determined based off the relevance
    of each word's score in the query. If pagerank is True, then the pagerank
    will also be factored into the score calculation. Next, the dictionary is
    sorted based off the pagerank values, using a helper method. Lastly,
    the first 10 results are printed. If fewer than 10 documents contain 
    the query, then only those documents are printed. If no documnets
    contain the word, then an error message is displayed. 
    '''
    def answer_query(self, input: List[str]):
        relevance_dict = {}
        for id in self.titles_dict:
            score = sum([self.get_relevance(id, word) for word in input])
            if self.pagerank:
                score*=self.docs_dict[id]
            relevance_dict[id] = score

        #sort the dictionary from high to low
        sorted_by_relevance = sorted(relevance_dict.items(), reverse=True, \
            key=self.sort_helper)

        #print first 10 
        for i in range(10):
            if sorted_by_relevance[i][1] > 0.0:
                print(self.titles_dict[sorted_by_relevance[i][0]])
            elif i==0 and sorted_by_relevance[i][1] == 0.0:
                print("No documents contain your input word(s); nothing to \
                     display.")
                break

    '''
    This is a helper method when sorting the dictionary of ids->relevances.
    It sorts based on the second item of the tuple, which is the value in the
    KV pair (relevance).
    '''
    def sort_helper(self, kv_pair: tuple) -> float:
        return kv_pair[1]

    '''
    Returns the relevance of a word for a given document. If the word
    appears in the document at least once, the relevance is returned. Else
    the KeyError is caught and 0.0 is returned to signify that the relevance
    is zero.
    '''
    def get_relevance(self, id: int, word: str) -> float:
        try:
            return self.words_dict[word][id]
        #if word not found in document
        except KeyError:
            return 0.0

    '''
    Processes the user input to be ready for querying. Any stop words are
    removed, and the words are stemmed. The words are also separated using
    a regex that separates words. A list containing each processed word
    is returned.
    '''
    def process_input(self, input: str) -> List[str]:
        STOP_WORDS = set(stopwords.words('english'))
        nltk_test = PorterStemmer()

        list_words = []

        text_regex = '''[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        words = re.findall(text_regex, input.lower())

        for word in words:
            if word not in STOP_WORDS:
                list_words.append(nltk_test.stem(word))
    
        return list_words

'''
The main method processes arguments input by the user. If fewer than 3 or
more than 4 arguments are inputted, an error message is printed and the program
stops running. If 3 arguments are inputted, then the Querier is instantiated
with the self.pagerank argument as False. If 4 arguments are inputted and the 
first argument is exactly --pagerank, then Querier is instantiated with pagerank
as True. Else, a different error message is printed. 
'''
if __name__ == '__main__':
    if len(sys.argv) - 1 < 3 or len(sys.argv) - 1 > 4:
        print('Wrong number of arguments. Please try again.')
        sys.exit
    #no --pagerank
    elif len(sys.argv) - 1 == 3:
        i = Querier(False, sys.argv[1], sys.argv[2], sys.argv[3])
    #include --pagerank
    elif len(sys.argv) - 1 == 4 and sys.argv[1] == '--pagerank':
        i = Querier(True, sys.argv[2], sys.argv[3], sys.argv[4])
    #case for if --pagerank argument is spelled wrong or otherwise incorrect
    else:
        print('Incorrect inputs. Please make sure inputs are correct.')
        sys.exit

