import urllib.request
import re
from yahoo_finance import Share

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

 
yahoo = Share('YHOO')
print(yahoo.get_open())
print(yahoo.get_price())
print(yahoo.get_trade_datetime())
