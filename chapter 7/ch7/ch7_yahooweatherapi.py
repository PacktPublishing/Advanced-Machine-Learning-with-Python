import urllib, json

baseurl = 'https://query.yahooapis.com/v1/public/yql?'

yql_query = "select item.condition from weather.forecast where woeid=9807"
yql_url = baseurl + urllib.urlencode({'q':yql_query}) + "&format=json"
result = urllib.urlopen(yql_url).read()
data = json.loads(result)
print data['query']['results']
