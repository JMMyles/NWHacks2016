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
	index_year = index.get_historical(year+'-01-01', year+'-12-31')
	s_open = float(index_year[0]['Open'])
	s_close = float(index_year[0]['Close'])
	s_open365 = float(index_year[len(index_year) - 1]['Open'])
	s_close365 = float(index_year[len(index_year) - 1]['Close'])
	s_open_diff = 1 + (s_open - s_open365) / s_open
	s_close_diff = 1 + (s_close - s_close365) / s_close

	return(s_open_diff,s_close_diff)

yahoo = Share('YHOO')
#print(yahoo.get_open())
#print(yahoo.get_price())
#print(yahoo.get_trade_datetime())

NASDAQ = Share('^NQDXUSB')
#print(NASDAQ.get_price())
#print(NASDAQ.get_change())
Dow = Share('^DJI')
#print(Dow.get_price())
sp500 = Share('^GSPC')
#print(sp500.get_price())
naught_open,naught_close = get_year_price('2000', Dow)
eight_open,eight_close = get_year_price('2008', Dow)

print(naught_close,naught_open,eight_close,eight_open)