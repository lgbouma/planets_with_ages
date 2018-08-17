'''
https://mast.stsci.edu/api/v0/MastApiTutorial.html
    and
https://mast.stsci.edu/api/v0/pyex.html
'''

import sys, os, time, re, json

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

def tic_single_object_crossmatch(ra, dec, radius):
    '''
    ra, dec, radius: all in decimal degrees

    speed tests: about 10 crossmatches per second.
        (-> 3 hours for 10^5 objects to crossmatch).
    '''

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":ra,"dec":dec}]}

    request =  {"service":"Mast.Tic.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":radius
                },
                "format":"json",
                'removecache':True}

    headers,outString = mast_query(request)

    outData = json.loads(outString)

    return outData


def mast_query(request):

    server='mast.stsci.edu'

    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "Connection": "close",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    del conn # NOTE: unclear why, but this is needed.

    return head,content


def tic_advanced_filter_search():
    request = {"service":"Mast.Catalogs.Filtered.Tic",
               "format":"json",
               "params":{
                   "columns":"c.*",
                   "filters":[
                       {"paramName":"dec",
                        "values":[{"min":86.,"max":88.}]},
                       {"paramName":"Teff",
                        "values":[{"min":5000.,"max":6000.}]},
                       {"paramName":"Tmag",
                        "values":[{"min":8.,"max":10.}]}]
               }}

    headers,outString = mast_query(request)
    outData = json.loads(outString)

    return outData


def tic_advanced_search_position():
    request = {"service":"Mast.Catalogs.Filtered.Tic.Position",
               "format":"json",
               "params":{
                   "columns":"c.*",
                   "filters":[
                       {"paramName":"Teff",
                        "values":[{"min":4250.,"max":4500.}]},
                       {"paramName":"logg",
                        "values":[{"min":4.4,"max":5.0}]},
                       {"paramName":"Tmag",
                        "values":[{"min":8.,"max":10.}]}],
                   "ra": 210.8023,
                   "dec": 54.349,
                   "radius": .2
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData

