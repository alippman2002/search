import pytest
import index
import query
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def test_tf_wiki_dicts():
    i = index.Indexer("wikis/test_tf_idf.xml", "titles.txt", "docs.txt", "words.txt")
    assert i.title_dict == {1: "page 1", 2: "page 2", 3: "page 3"}
    assert len(i.title_dict) == 3
    assert len(i.count_dict) == 3
    assert len(i.count_dict[1]) == 5
    assert len(i.links_dict) == 3

def test_tf_tf_xml():
    i = index.Indexer("wikis/test_tf_idf.xml", "titles.txt", "docs.txt", "words.txt")
    assert i.calculate_term_frequency() == \
        {1: {'bit': 1.0, 'dog': 1.0, 'man': 1.0, 'page': 1.0, '1': 1},
        2: {'ate': 1.0, 'chees': 1.0, 'dog': 1.0, 'page': 1.0, '2': 1},
        3: {'bit': 0.5, 'chees': 1.0, 'page': 0.5, '3': 0.5}} 

def test_tr_tf_xml():
    i = index.Indexer("wikis/test_tf_idf.xml", "titles.txt", "docs.txt", "words.txt")
    #test term relevance through words dict, which contains it
    #word -> {id: tr}
    #calculations done by hand
    assert i.words_dict == \
        {'bit': {1: pytest.approx(0.405, 0.01), 3: pytest.approx(0.203, 0.01)},
        'dog': {1: pytest.approx(0.405, 0.01), 2: pytest.approx(0.405, 0.01)},
        'man': {1: pytest.approx(1.099, 0.01)},
        'ate': {2: pytest.approx(1.099, 0.01)},
        'chees': {2: pytest.approx(0.405, 0.01), 3: pytest.approx(0.405, 0.01)},
        'page': {1: 0.0, 2: 0.0, 3: 0.0},
        '1': {1: pytest.approx(1.099, 0.01)},
        '2': {2: pytest.approx(1.099, 0.01)},
        '3': {3: pytest.approx(0.545, 0.01)}}

def test_empty_wiki():
    i = index.Indexer("wikis/empty_wiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert i.docs_dict == {}
    assert i.links_dict == {}
    assert i.words_dict == {}

def test_pagerank():
    i = index.Indexer("wikis/PageRankWiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)
    assert i.docs_dict[100] > i.docs_dict[1]

def test_small_added():
    i = index.Indexer("wikis/test_small_wiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)

def test_small_one_by_one():
    i = index.Indexer("wikis/test_wiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)

'''
def test_pagerank_smallwiki():
    i = index.Indexer("wikis/SmallWiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)

def test_pagerank_medwiki():
    i = index.Indexer("wikis/MedWiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)

def test_pagerank_bigwiki():
    i = index.Indexer("wikis/BigWiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)
'''

def test_title():
    i = index.Indexer("wikis/title_wiki.xml", "titles.txt", "docs.txt", "words.txt")
    assert i.title_dict == {2: "egg"}


def test_pagerank_1():
    #Expected: Rank(A) = 0.4326, Rank(B) = 0.2340, Rank(C) = 0.3333
    i = index.Indexer("wikis/PageRankExample1.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)
    assert i.docs_dict[1] == pytest.approx(0.4326, 0.001)
    assert i.docs_dict[2] == pytest.approx(0.2340, 0.001)
    assert i.docs_dict[3] == pytest.approx(0.3333, 0.001)

def test_pagerank_2():
    #Expected: Rank(A) = 0.2018, Rank(B) = 0.0375, Rank(C) = 0.3740, Rank(D) = 0.3867
    i = index.Indexer("wikis/PageRankExample2.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)
    assert i.docs_dict[1] == pytest.approx(0.2018, 0.001)
    assert i.docs_dict[2] == pytest.approx(0.0375, 0.001)
    assert i.docs_dict[3] == pytest.approx(0.3740, 0.001)
    assert i.docs_dict[4] == pytest.approx(0.3867, 0.001)

def test_pagerank_3():
    #Expected: Rank(A) = 0.0524, Rank(B) = 0.0524, Rank(C) = 0.4476, Rank(D) = 0.4476
    i = index.Indexer("wikis/PageRankExample3.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)
    assert i.docs_dict[1] == pytest.approx(0.0524, 0.001)
    assert i.docs_dict[2] == pytest.approx(0.0524, 0.001)
    assert i.docs_dict[3] == pytest.approx(0.4476, 0.001)
    assert i.docs_dict[4] == pytest.approx(0.4476, 0.001)

def test_pagerank_4():
    #<Expected: Rank(A) = 0.0375, Rank(B) = 0.0375, Rank(C) = 0.4625, Rank(D) = 0.4625
    i = index.Indexer("wikis/PageRankExample4.xml", "titles.txt", "docs.txt", "words.txt")
    assert sum(i.docs_dict.values()) == pytest.approx(1, 0.001)
    assert i.docs_dict[1] == pytest.approx(0.0375, 0.001)
    assert i.docs_dict[2] == pytest.approx(0.0375, 0.001)
    assert i.docs_dict[3] == pytest.approx(0.4625, 0.001)
    assert i.docs_dict[4] == pytest.approx(0.4625, 0.001)

def test_system1():
    # System testing: using Medwiki to test index and query (No pagerank)
    a = index.Indexer("wikis/MedWiki.xml", "titles.txt", "docs.txt", "words.txt")
    b = query.Querier(False, "titles.txt", "docs.txt", "words.txt")
    user_input = "baseball"
    # Does this output the corect amount of pages
    assert len(b.answer_query(b.process_input(user_input))) == 10
    # Confirms output documents are reasonable results
    print(b.answer_query)

def test_system2():
    # System testing: using Medwiki to test index and query (Yes pagerank)
    a = index.Indexer("wikis/MedWiki.xml", "titles.txt", "docs.txt", "words.txt")
    b = query.Querier(True, "titles.txt", "docs.txt", "words.txt")
    user_input = "baseball"
    # Does this output the corect amount of pages
    assert len(b.answer_query(b.process_input(user_input))) == 10
    # Confirms output documents are reasonable results
    print(b.answer_query)
