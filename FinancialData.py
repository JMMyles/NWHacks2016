import urllib.request
import re
from yahoo_finance import Share

#def get_stock_quote(symbol):
#	base_url = 'http://finance.google.com/finance?q='
#	#base_url = 'https://www.google.com/finance?q'
#	content = urllib.request.urlopen(base_url + symbol).read()
#	m = re.search(b'id="ref_694653_l".*?>(.*?)<', content)
#	if m:
#		quote = m.group(1)
#	else:
#		quote = 'no quote available for: ' + symbol
#	return quote

#print(get_stock_quote('INDEXNASDAQ:.IXIC'))
#print(get_stock_quote('INDEXDJX:.DJI'))
#print(get_stock_quote('INDEXSP:.INX'))
#print(get_stock_quote('NAHB')) # Housing market index
#print(get_stock_quote('gool'))
