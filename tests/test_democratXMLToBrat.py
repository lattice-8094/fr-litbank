import pytest
from lxml import etree

import sys
sys.path.append('../src')
import DemocratXMLToBrat

@pytest.fixture
def input_data():
    xml_tree = etree.parse('test_data.xml')
    xml_root = xml_tree.getroot()
    return xml_root

@pytest.fixture
def compute_text(input_data):
    sentences = DemocratXMLToBrat.get_sentences(input_data)
    with open('test.txt', 'w') as txt:
        for s in sentences:
            print(s, end="", file=txt)
    brat_txt = ""
    with open('test.txt') as txt:
        brat_txt = txt.read()
    return brat_txt

def test_first_word(input_data, compute_text):
    """
    Test against the generated brat file that the first word of each sentence has the correct indexes
    """
    sentences = DemocratXMLToBrat.get_sentences(input_data)
    for s in sentences:
        first_word = s.content[0]
        assert first_word.printable == compute_text[first_word.start:first_word.get_end()]

def test_3last_word(input_data, compute_text):
    """
    Test against the generated brat file that the last -3 word of each sentence has the correct indexes
    """
    sentences = DemocratXMLToBrat.get_sentences(input_data)
    for s in sentences:
        first_word = s.content[-3]
        assert first_word.printable == compute_text[first_word.start:first_word.get_end()]