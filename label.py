from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.stem.porter import *
import numpy as np
from datetime import datetime
from datetime import timedelta

class Label:

	def __init__(self, path):
		self.path = path
		self.DELTA = timedelta(minutes=60)

	def parse_timestamp(self, timestamp):
		return datetime.strptime(timestamp, '%A, %B %d, %Y at %I:%M%p %Z')

	def label_messages(self, messages, top_n):
		stemmer = PorterStemmer()
		corpus = []
		for message in messages:
			stemmed = []
			for word in message.text.split(" "):
				stemmed.append(stemmer.stem(word))
			corpus.append(' '.join(stemmed))
		tfidf = TfidfVectorizer(tokenizer = None, stop_words = 'english')
		tfidf.fit_transform(corpus)
		indices = np.argsort(tfidf.idf_)[::-1]
		features = tfidf.get_feature_names()
		labels = [features[i] for i in indices[:top_n]]
		return labels

	def scrape(self):
		with open(self.path) as f:
			messages = []
			for line in f:
				soup = bs(line, 'html.parser')
				times = soup.findAll("div", attrs={'class' : 'message'})
				msgs = soup.findAll('p')
				for time, msg in zip(times, msgs):
					msg_info = time.findAll('span')
					msg_text = msg.getText()
					sender = msg_info[0].getText()
					timestamp = self.parse_timestamp(msg_info[1].getText()[:-3]) # strip UTC offset
					message = Message(msg_text, timestamp, sender)
					messages.append(message)
					if len(messages) > 2 and messages[-2].timestamp - messages[-1].timestamp > self.DELTA:
						for message in reversed(messages):
							print(message.sender.encode('utf-8') + str(message.timestamp).encode('utf-8'))
							print(message.text.encode('utf-8'))
						labels = self.label_messages(messages, 10)
						print(str(labels))
						messages = []
#						break

		 

class Message:

	def __init__(self, text, timestamp, sender):
		self.text = text
		self.timestamp = timestamp
		self.sender = sender


# To run
m = LabelMe('html/messages.htm')
m.scrape()