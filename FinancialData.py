import urllib.request
import re
from yahoo_finance import Share

#def get_stock_quote(symbol):
#	base_url = 'http://finance.google.com/finance?q='
#	content = urllib.request.urlopen(base_url + symbol).read()
#	m = re.search(b'id="ref_694653_l".*?>(.*?)<', content)
#	if m:
#		quote = m.group(1)
#	else:
#		quote = 'no quote available for: ' + symbol
#	return quote

#print(get_stock_quote('NASDAQ'))


def get_year_price(year, index):
	index_year = index.get_historical(year+'01-01', year+'12-31')
	price = 0
	for stock in index_year:
		price += stock.price
		price /= 2

yahoo = Share('YHOO')
print(yahoo.get_open())
print(yahoo.get_price())
print(yahoo.get_trade_datetime())
print(yahoo.get_historical('2008-01-01', '2008-01-31'))
#print(yahoo.get_historical('2008-01-01', '2008-12-31'))
NASDAQ = Share('^NQDXUSB')
print(NASDAQ.get_price())
print(NASDAQ.get_change())
Dow = Share('^DJI')
print(Dow.get_price())
sp500 = Share('^GSPC')
print(sp500.get_price())
print(get_year_price('2000', NASDAQ))

