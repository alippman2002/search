import xml.etree.ElementTree as et
import sys
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List
from typing import Dict
import file_io
import math

""" Documentation for the constructor:
    The init method is the constructor for the indexer class.
    Parameters: self, xml, a string representation of the filepath of the
    xml wiki file being parsed to be indexed and queried, title, a string
    representation of the filepath for titles.txt (where titles are 
    written to), docs, a string representation of the filepath for docs.txt
    (where pagerrank is written to), and words, a string representation of
    the filepath for words.txt (where words and term relevance is written to
    for every ID)
    Returns: NA (just setsup data structures and parses through the xml file)
    Throws: NA
"""
class Indexer:
    def __init__(self, xml: str, title: str, docs: str, words: str):
        self.xml_filepath = xml
        self.titles_filepath = title
        self.docs_filepath = docs
        self.words_filepath = words
        
        self.n = 0

        self.title_dict = {}
        self.docs_dict = {}
        self.words_dict = {}
        self.links_dict = {}
        self.count_dict = {}

        self.xml_parser()
        self.calculate_term_relevance()
        self.pagerank()
        
    """ Documentation for XML parser:
        This method parses an XML file and sets up all of the required
        global variable data structures needed for Indexing and Querying
        a search.
        Parameters: none (just inputs self)
        Returns: NA (just parses XML file)
        Throws: NA
        """
    def xml_parser(self):
        root: et.Element = et.parse(self.xml_filepath).getroot()
        all_pages: et.ElementTree = root.findall('page')
        for page in all_pages:
            #populate title dict with id -> title
            title: str = page.find('title').text.strip().lower()
            id: int = int(page.find('id').text)
            self.title_dict[id] = title

            #populate doc dict with id -> none
            self.docs_dict[id] = None
            self.links_dict[id] = []

            text: str = page.find('text').text
            text+=title
            self.process_text(text, id)
        
        self.n = len(self.links_dict)
        
        #write to title file
        file_io.write_title_file(self.titles_filepath, self.title_dict)

    """ Documentation for process_text:
        This method inputs all of the text located in a wiki file and processes
        it by making it lowercase, identifying words and links using regex
        and removing stop words and stemming words and setting up the
        corresponding data structures for this data.
        Parameters: self, txt, a string which is all of the text that is within 
        the <text> </text> portion of the XML file meant to be processed, and
        id, a integer which corresponds to the ID of the document whose
        text is being processed.
        Returns: NA (just processes all text)
        Throws: NA
        """
    def process_text(self, text: str, id: int):
        #all text
        text = text.lower()
        full_regex = '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        tokens = re.findall(full_regex, text)

        #just link regex
        link_regex = '''\[\[[^\[]+?\]\]'''
        links = re.findall(link_regex, text)
        
        #stopwords and stemming
        STOP_WORDS = set(stopwords.words('english'))
        nltk_test = PorterStemmer()

        #dict that will hold counts of words for page
        page_count_dict = {}

        #loop through words in list of tokens. don't consider stop words
        for word in tokens:
            if word not in STOP_WORDS:
                if word in links:
                    #append words in link that should be processed to list of tokens
                    words_to_append = self.process_links(word, id)
                    for each_word in words_to_append:
                        tokens.append(each_word)
                
                else: 
                    #stem the word
                    stemmed_word = nltk_test.stem(word)

                    #words dict (word --> dict of id --> tr)
                    if stemmed_word not in self.words_dict:
                        self.words_dict[stemmed_word] = {id: None}
                    else:
                        dict_with_id = self.words_dict.get(stemmed_word)
                        dict_with_id[id] = None
                    
                    #add word to list of counts
                    if stemmed_word not in page_count_dict:
                        page_count_dict[stemmed_word] = 1
                    else:
                        page_count_dict[stemmed_word]+=1
        
        #when done iterating through tokens
        self.count_dict[id] = page_count_dict

    """ Documentation for process_links: 
        This method processses all words that have been identified as links 
        to put them into their corresponding data structures for indexing and
        querying. 
        Parameters: self, link, a string representation of a link, and id, an 
        integer corresponding to the id the link was found in.
        Returns: a list of links that must be appended to tokens to be procesed
        in process_text
        Throws: NA
        """
    def process_links(self, link: str, id: int) -> List[str]:
        #take brackets out of link
        link_no_brackets = link[2:-2]

        if '|' in link_no_brackets:
            link_word = link_no_brackets.split('|')[0]
            page_word = link_no_brackets.split('|')[1]
        else:
            link_word = link_no_brackets
            page_word = link_no_brackets

        if link_word not in self.links_dict[id]:
            self.links_dict[id].append(link_word)

        #separate into each word, then append to list of words to process
        return self.split_into_words(page_word)
    
    """ Documentation for split_into_words:
        This method uses a regex to find all words and split them up.
        Parameters: self, words, a string representation of a passed in word
        that may be multiple words and gets rid of , : to parse text correctly
        Returns: a list of strings corresponding to all of the viable strings
        found in words with the text_regex
        Throws: NA
        """
    def split_into_words(self, words: str) -> List[str]:
        text_regex = '''[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        return re.findall(text_regex, words)

    """ Documentation for calculate_term_relevance:
        This method calculates term relevance for every word in the corpus
        for every ID by calling helpers to calculate term frequency and inverse
        doc frequency.
        Parameters: self
        Returns: NA (just updates the term relevance dictionary)
        Throws: NA
        """
    def calculate_term_relevance(self):
        #tf dict is id --> {word, tf}
        dict_with_tf = self.calculate_term_frequency()

        #idf dict is word --> idf
        dict_with_idf = self.calculate_inverse_doc_frequency()

        self.term_relevance(dict_with_tf, dict_with_idf)

    """ Documentation for term relevance:
        This method utilizes the tf and idf from their respective dictionaries
        and multiplies them to find the term relevance of every word in the 
        corpus (for every ID a word is in)
        Parameters: self, tf_dict, a dictionary that maps ID to a dictionary
        of words to float (representing tf), and idf_dict, a dictionary that
        maps a word to its IDF.
        Returns: NA (just updates term relevance)
        Throws: NA
        """
    def term_relevance(self, tf_dict, idf_dict):
        #for each word
        for word in self.words_dict:
            #for each id
            for id in self.words_dict[word]:
                tf = tf_dict[id][word]
                idf = idf_dict[word]
                #calculate tr
                tr = tf * idf
                #add to value in sub dict
                self.words_dict[word][id] = tr

        #write to words file
        file_io.write_words_file(self.words_filepath, self.words_dict)

    """ Documentation for calculate_term_frequency:
        This method fills the dict_id_to_tf dictionary with all Ids of 
        all documents in the wiki as keys to a subdictionary that maps
        all words to a count for their frequency in the document. 
        Parameters: none (just inputs self)
        Returns: dict_id_to_tf (the term frequency dictionary as described 
        above)
        Throws: NA
        """
    def calculate_term_frequency(self) -> Dict[int, Dict[str, float]]:
        #dict to return
        dict_id_to_tf = {}

        #iterate through each id in count dict
        for id in self.count_dict:
            #create local dict to add later
            dict_id_to_tf_sub = {}
            #access sub dictionary word --> count
            sub_dict = self.count_dict[id]
            
            if sub_dict:
                #get a (max count)
                max_count = max(sub_dict.values())
                #iterate through words in sub dict
                for word in sub_dict:
                    #get c (count of word)
                    count_word = sub_dict[word]
                    #find tf
                    tf = float(count_word)/max_count
                    #populate sub dict for tf
                    dict_id_to_tf_sub[word] = tf
                #once done with all words, add to tf dict
                dict_id_to_tf[id] = dict_id_to_tf_sub
        return dict_id_to_tf
    
    """ Documentation for calcualte_inverse_doc_frequency: 
        This method calculates the IDF by taking the log of how many docs
        there are divided by the instanced of documents with the word in it
        Parameters: none (just inputs self)
        Returns: dict_idf, a inverse document frequency dictionary
        that maps words as keys to the idf value found
        Throws: NA"""
    def calculate_inverse_doc_frequency(self) -> Dict[str, float]:
        #dict to return
        dict_idf = {}

        #find ni per word
        for word in self.words_dict:
            ni = float(len(self.words_dict[word]))
            idf = math.log(self.n/ni)
            dict_idf[word] = idf
        
        return dict_idf

    """ Documentation for pagerank:
        This method calculates the pagerank values for every document within the 
        passed in wiki in self to determine the authoritativeness of docs.
        Parameters: none (just inputs self)
        Returns:  NA (just calculates pagerank values)
        Throws: NA"""
    def pagerank(self):
        #dict of (id, linked_id) -> weight
        weights_dict = self.populate_weights_dict()

        #initialize every rank in r to be 0
        #initialize every rank in r' to be 1/n
        r = {}
        r_prime = {}
        for id in self.title_dict:
            r[id] = 0
            r_prime[id] = 1/self.n
        

        while self.distance(r, r_prime) > 0.001:
            r = r_prime.copy()
            for j in self.docs_dict:
                r_prime[j] = 0
                for k in self.docs_dict:
                    r_prime[j] = r_prime[j] + weights_dict[(k, j)] * r[k]

        #when done, populate docs_dict
        self.docs_dict = r_prime
        file_io.write_docs_file(self.docs_filepath, self.docs_dict)

    """ Documentation for distance:
        This method is used to check for convergence of r and r prime to see if
        the iterations of values found for pagerank have converged or not
        Parameters: self, r, a dictionary that maps a id key to a float of its
        pagerank value and r' a dictionary that is the same but for the next
        iteration of the pagerank algorithm. 
        Returns: a float value describing the euclidian distance between 
        the iterations of r and r prime.
        Throws: NA"""
    def distance(self, r: Dict[int, float], r_prime: Dict[int, float]) -> float:
        sum = 0
        for id in r:
            sum += ((r_prime[id] - r[id])*(r_prime[id] - r[id]))
        
        return math.sqrt(sum)
        

    """ Documentation for populate_weights_dict:
        This method calculates all of the weights for every page in the wiki,
        which is done once initially at the beginning of implementing the 
        pagerank algorithm.
        Parameters: none (just inputs self)
        Returns: weights_dict, a dictionary that maps tuples of (id (start link)
        to id (end link)) as keys to represent edges and the float is the weight
        value associated with the edge that has been computed.
        Throws: NA
        """
    def populate_weights_dict(self) -> Dict[tuple, float]:
        #reverse id to title dict. title -> idf
        title_to_id = {title:id for id, title in self.title_dict.items()}
        
        #(id, id) -> weight
        weights_dict = {}

        #for each id in the doc
        for id in self.docs_dict:
            #list of unique links that id links to IN THE CORPUS
            links = [link for link in self.links_dict[id] if link in title_to_id]

            #if the page links to nothing or only itself
            if (len(links) == 1 and title_to_id[links[0]] == id) or (not links):
                for title in title_to_id:
                    curr_id = title_to_id[title]
                    if id == curr_id:
                        weights_dict[(id, curr_id)] = 0.15/self.n
                    else:
                        weights_dict[(id, curr_id)] = 0.15/self.n + 0.85*(1/(self.n-1))
            else:
                for title in title_to_id:
                    curr_id = title_to_id[title]
                    if title in links and curr_id != id:
                        weights_dict[(id, curr_id)] = 0.15/self.n + 0.85*(1/len(links))
                    else:
                        weights_dict[(id, curr_id)] = 0.15/self.n

        return weights_dict
        
""" Documentation for main method:
    This method ensures the proper number of arguments are being input (which
    should be 4 and exits if any other number of arguments are passed in. If
    4 are passed in the indexer uses these arguments and is ran!"""                
if __name__ == '__main__':
    if len(sys.argv) - 1 != 4:
        print('Wrong number of arguments. Please try again.')
        sys.exit
    else:
        i = Indexer(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])