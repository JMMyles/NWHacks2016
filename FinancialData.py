import urllib.request
import re

def get_stock_quote(symbol):
	base_url = 'http://finance.google.com/finance?q='
	content = urllib.request.urlopen(base_url + symbol).read()
	m = re.search(b'id="ref_694653_l".*?>(.*?)<', content)
	if m:
		quote = m.group(1)
	else:
		quote = 'no quote available for: ' + symbol
	return quote

print(get_stock_quote('NASDAQ'))
