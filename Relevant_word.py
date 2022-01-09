def relevant_words(links, manual='yes', html_conversion='html.parser', eng_selection='nltk', documentation='no'):
    
    #doucmentation to describe usage
    if documentation == 'yes':
        return(print("relevant_words(links, manual='yes', html_conversion='html.parser', eng_selection='nltk', documentation='no')", \
                     "\n\n links (array): Documents to be scanned and to find relevant words of.", \
                     "\n\n manual (str): {'yes','no'} Determines the term frequency inverse document frequency version to be used. Manually written by me, other alternative by Sklearn.", \
                     "\n\n html_conversion (str): {'html.parser', 'lxml', 'lxml-xml', 'xml', 'html5lib'} Determines method of html parser for bs4 from BeautifulSoup. ref https://www.crummy.com/software/BeautifulSoup/bs4/doc/", \
                     "\n\n eng_selection (str): {'nltk', 'enchant', 'both'} Determines the method for filtering words.",\
                     "\n\t nltk - Uses wordnet database of English. ref https://wordnet.princeton.edu/",\
                     "\n\t enchant - Uses spellchecking library by pyenchant. ref https://pypi.org/project/pyenchant/",\
                     "\n\t both - Uses both wordnet and enchant.",\
                     "\n\n documentation (str): {'yes','no'} Used to display documentation."))
    
    #imports
    import numpy as np
    import json
    import math
    import urllib
    from bs4 import BeautifulSoup
    import enchant
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet as wn
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    
    #list of acceptable english characters
    def map_non_alphabet(letter):
        alphabet = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','y','x','z',"'"]
        if letter in alphabet:
            return letter
        else:
            return ' '
    
    #extracts words from a website and saves as string
    def website_text(link, html_conversion):

        f = urllib.request.urlopen(link)
        file = f.read()

        text = BeautifulSoup(file, html_conversion).get_text() #html to text

        lt = text.lower()
        lt = lt.replace("\n", " ")

        text = ''.join(map(map_non_alphabet, lt)) #removes all non alphabet, space and ' characters (' needed for words like don't)

        #remove all double spacing
        while("  " in text):
            text = text.replace("  ", " ")

        return(text)
    
    #frequency update function
    def frequency_update(dictionary, val):
        if val in dictionary:
            dictionary[val] += 1
        else:
            dictionary[val] = 1
            
    #function extracts         
    def words_list(text, eng_selection):
        list_text = text.split(' ')

        #method for checking if word is english uses enchant dictionary, nltk wordnet or both
        d = enchant.Dict('en')
        eng_check = lambda x: wn.synsets(x) if eng_selection=='nltk' else \
        (d.check(x) if eng_selection=='enchant' else wn.synsets(x) and d.check(x))

        freq = {} #dictionary to hold the frequency of each word in the website and total number of words
        for i in list_text:

            #remove blanks
            if i == '':
                continue

            #Removes apostrophies at start or end as most likely used as quotation marks
            if i[0] == "'":
                i = i[1:]
            if i == '': #check to see if makes string empty
                continue
            if i[-1] == "'":
                i = i[:-1]
            if i == '': #check to see if makes string empty
                continue

            if len(i) == 1:
                if i == 'a' or i == 'i': #only single letter words are a and i (removes intials)
                    frequency_update(freq, i)
                else:
                    continue

            elif eng_check(i): #word in English Dictionary
                frequency_update(freq, i)

        return(freq, sum(freq.values()))
    
    if manual == 'yes':
        
        total_docs = len(links)
        to_words = {} #dictionary for frequency of word occurance for each link
        for link in links:
            text = website_text(link, html_conversion)
            to_words[link] = words_list(text, eng_selection)

        words_doc_occurances = {} #dictionary listing all words and the number of documents they occur in

        for i in to_words:
            for j in to_words[i][0]:
                frequency_update(words_doc_occurances, j)

        tfidf = {} #dictionary listing the term frequency inverse docuement frequency for each word within each link

        for i in to_words:
            tfidf[i] = {} #subdictionary for each link
            total_doc_words = to_words[i][1]
            for j in to_words[i][0]:
                tfidf[i][j] = (to_words[i][0][j]/total_doc_words) * math.log(total_docs/words_doc_occurances[j])

        out = {} #dictionary to be output as json file
        for i in tfidf:
            nd = tfidf[i]
            sorted_freq = sorted(nd.items(), key = lambda nd: nd[1], reverse=True)
            out[i] = sorted_freq[0][0:20]
            w = [] #hold only word ignore tfidf score
            for j in sorted_freq[0:20]:
                w.append(j[0])
            out[i] = w  
        
    else:
        
        documents= [] #list to hold all website text as strings

        for l in links:
            documents.append(website_text(l, html_conversion))

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_vectorizer.fit(documents)

        out = {} #output dictionary

        #Attapted implemtation from https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
        for i in range(len(links)):
            tfidf = tfidf_vectorizer.transform(documents[i].split(' '))
            importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
            tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())
            out[links[i]] =  list(tfidf_feature_names[importance[:20]])

    #print(json.dumps(out, indent=4))
    with open('relevant_words.json', 'w') as w:
        json.dump(out, w, indent=4)
    
