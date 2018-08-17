#!/usr/bin/env python
'''
tools copied from

https://mast.stsci.edu/api/v0/MastApiTutorial.html
    and
https://mast.stsci.edu/api/v0/pyex.html
'''

## [Includes]
import sys
import os
import time
import re
import json

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
## [Includes]

## [Mast Query]
def mastQuery(request):

    server='mast.stsci.edu'

    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
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

    return head,content
## [Mast Query]


## [Download Request]
def downloadRequest(url):
    server='mast.stsci.edu'

    conn = httplib.HTTPSConnection(server)
    conn.request("GET", url)
    resp = conn.getresponse()

    fileName = resp.getheader('Content-Disposition')[21:]
    fileContent = resp.read()

    with open(fileName,'wb') as FLE:
        FLE.write(fileContent)

    conn.close()

    return fileName
## [Download Request]    


## [Json to csv]
def mastJson2Csv(json):
    csvStr =  ",".join([x['name'] for x in json['fields']])
    csvStr += "\n"
    csvStr += ",".join([x['type'] for x in json['fields']])
    csvStr += "\n"

    colNames = [x['name'] for x in json['fields']]
    for row in json['data']:
        csvStr += ",".join([str(row.get(col,"nul")) for col in colNames]) + "\n"

    return csvStr
## [Json to csv]


## [Json to astropy]
from astropy.table import Table
import numpy as np

def mastJson2Table(jsonObj):

    dataTable = Table()

    for col,atype in [(x['name'],x['type']) for x in jsonObj['fields']]:
        if atype=="string":
            atype="str"
        if atype=="boolean":
            atype="bool"
        dataTable[col] = np.array([x.get(col,None) for x in jsonObj['data']],dtype=atype)

    return dataTable
## [Json to astropy]


## [Name Resolver]
def resolveName():

    request = {'service':'Mast.Name.Lookup',
               'params':{'input':'M101',
                         'format':'json'},
    }

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [Name Resolver]

## [List Missions]
def listCaomMissions():

    request = {
        'service':'Mast.Missions.List',
        'params':{},
        'format':'json'
    }

    headers,outString = mastQuery(request)

    outData = [x['distinctValue'] for x in json.loads(outString)['data']]

    return outData
## [List Missions]

## [CAOM Cone Search]    
def caomConeSearch():

    request = {'service':'Mast.Caom.Cone',
               'params':{'ra':254.28746,
                         'dec':-4.09933,
                         'radius':0.2},
               'format':'json',
               'pagesize':5000,
               'page':1,
               'removenullcolumns':True,
               'timeout':30,
               'removecache':True}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [CAOM Cone Search]


## [VO Cone Search]
def voConeSearch():

    request = {'service':'Vo.Hesarc.DatascopeListable',
               'params':{'ra':254.28746,
                         'dec':-4.09933,
                         'radius':0.2,
                         'skipcache':True},
               'format':'json',
               'removenullcolumns':True}

    allData = []
    startTime = time.time()

    while True:
        headers,outString = mastQuery(request)
        outData = json.loads(outString)
        allData.append(outData)
        if outData['status'] != "EXECUTING":
            break
        if time.time() - startTime > 30:
            print("Working...")
            startTime = time.time()
        time.sleep(10)

    return allData
## [VO Cone Search]

## [HSC V2 Cone Search]
def hscV2ConeSearch():

    request = {'service':'Mast.Hsc.Db.v2',
               'params':{'ra':254.287,
                         'dec':-4.09933,
                         'radius':0.2,
                         'nr':5000,
                         'ni':1,
                         'magtype':1},
               'format':'json',
               'pagesize':1000,
               'page':1,
               'removenullcolumns':True}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [HSC V2 Cone Search]

## [HSC V3 Cone Search]
def hscV3ConeSearch():

    request = {'service':'Mast.Hsc.Db.v3',
               'params':{'ra':254.287,
                         'dec':-4.09933,
                         'radius':0.2,
                         'nr':5000,
                         'ni':1,
                         'magtype':1},
               'format':'json',
               'pagesize':1000,
               'page':1,
               'removenullcolumns':True}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [HSC V3 Cone Search]

## [GAIA DR1 Cone Search]
def gaiaDR1ConeSearch():
    request = {'service':'Mast.Catalogs.GaiaDR1.Cone',
               'params':{'ra':254.287,
                         'dec':-4.09933,
                         'radius':0.2},
               'format':'json',
               'pagesize':1000,
               'page':5}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [GAIA DR1 Cone Search]


### [GAIA DR2 Cone Search]
def gaiaDR2ConeSearch():
    request = {'service':'Mast.Catalogs.GaiaDR2.Cone',
               'params':{'ra':254.287,
                         'dec':-4.09933,
                         'radius':0.2},
               'format':'json',
               'pagesize':1000,
               'page':5}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [GAIA DR2 Cone Search]


## [TGAS Cone Search]
def tgasConeSearch():
    request = { "service":"Mast.Catalogs.Tgas.Cone",
                "params":{
                    "ra":254.28746,
                    "dec":-4.09933,
                    "radius":0.2},
                "format":"json",
                "timeout":10}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [TGAS Cone Search]

## [TIC Cone Search]
def ticConeSearch():
    request = { "service":"Mast.Catalogs.Tic.Cone",
                "params":{
                    "ra":254.28746,
                    "dec":-4.09933,
                    "radius":0.2},
                "format":"json",
                "timeout":10}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [TIC Cone Search]

## [TIC Advanced Search]
def ticAdvancedSearch():
    request = {"service":"Mast.Catalogs.Filtered.Tic",
               "format":"json",
               "params":{
                   "columns":"c.*",
                   "filters":[
                       {"paramName":"dec",
                        "values":[{"min":-90.,"max":-30.}]},
                       {"paramName":"Teff",
                        "values":[{"min":4250.,"max":4500.}]},
                       {"paramName":"logg",
                        "values":[{"min":4.4,"max":5.0}]},
                       {"paramName":"Tmag",
                        "values":[{"min":8.,"max":10.}]}]
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [TIC Advanced Search]

## [TIC Advanced Search Position]
def ticAdvancedSearchPosition():
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
## [TIC Advanced Search Position]

## [DD Cone Search]
def ddConeSearch():
    request = { "service":"Mast.Catalogs.DiskDetective.Cone",
                "params":{
                    "ra":254.28746,
                    "dec":-4.09933,
                    "radius":0.2},
                "format":"json",
                "timeout":10}   

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [DD Cone Search]

## [DD Advanced Search]
def ddAdvancedSearch():
    request = {"service":"Mast.Catalogs.Filtered.DiskDetective",
               "format":"json",
               "params":{
                   "filters":[
                       {"paramName":"classifiers",
                        "values":[{"min":10,"max":18}]},
                       {"paramName":"oval",
                        "values":[{"min":15,"max":76}]}]
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [DD Advanced Search]

## [DD Advanced Search Position]
def ddAdvancedSearchPosition():
    request = {"service":"Mast.Catalogs.Filtered.DiskDetective.Position",
               "format":"json",
               "params":{
                   "filters":[
                       {"paramName":"classifiers",
                        "values":[{"min":10,"max":18}]}],
                   "ra": 86.6909,
                   "dec": 0.079,
                   "radius": .2
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [DD Advanced Search Position]

## [DD Advanced Search Counts]
def ddAdvancedSearchCounts():
    request = {"service":"Mast.Catalogs.Filtered.DiskDetective.Count",
               "format":"json",
               "params":{
                   "filters":[
                       {"paramName":"classifiers",
                        "values":[{"min":10,"max":18}]},
                       {"paramName":"oval",
                        "values":[{"min":15,"max":76}]}]
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [DD Advanced Search Counts]

## [DD Advanced Search Position Counts]
def ddAdvancedSearchPositionCounts():
    request = {"service":"Mast.Catalogs.Filtered.DiskDetective.Position.Count",
               "format":"json",
               "params":{
                   "filters":[
                       {"paramName":"classifiers",
                        "values":[{"min":10,"max":18}]}],
                   "ra": 86.6909,
                   "dec": 0.079,
                   "radius": .2
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [DD Advanced Search Position Counts]

## [Advanced Search]
def advancedSearchCounts():
    request = {"service":"Mast.Caom.Filtered",
               "format":"json",
               "params":{
                   "columns":"COUNT_BIG(*)",
                   "filters":[
                       {"paramName":"filters",
                        "values":["NUV","FUV"],
                        "separator":";"
                       },
                       {"paramName":"t_max",
                        "values":[{"min":52264.4586,"max":54452.8914}], #MJD
                       },
                       {"paramName":"obsid",
                        "values":[],
                        "freeText":"%200%"}
                   ]}}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData


def advancedSearch():
    request = {"service":"Mast.Caom.Filtered",
               "format":"json",
               "params":{
                   "columns":"*",
                   "filters":[
                       {"paramName":"dataproduct_type",
                        "values":["image"],
                       },
                       {"paramName":"proposal_pi",
                        "values":["Osten"]
                       }
                   ],
                   "obstype":"all"
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [Advanced Search]


## [Advanced Search Position]
def advancedSearchWithPositionCounts():
    request = { "service":"Mast.Caom.Filtered.Position",
                "format":"json",
                "params":{
                    "columns":"COUNT_BIG(*)",
                    "filters":[
                        {"paramName":"dataproduct_type",
                         "values":["cube","image"]
                        }],
                    "position":"210.8023, 54.349, 5"
                }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData


def advancedSearchWithPosition():
    request = {"service":"Mast.Caom.Filtered.Position",
               "format":"json",
               "params":{
                   "columns":"*",
                   "filters":[
                       {"paramName":"dataproduct_type",
                        "values":["cube"]
                       }],
                   "position":"210.8023, 54.349, 0.24"
               }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData
## [Advanced Search Position]


## [HSC Spectra]
def hscSpectraSearch():

    request = {'service':'Mast.HscSpectra.Db.All',
               'format':'votable'}   

    headers,outString = mastQuery(request)

    return outString
## [HSC Spectra]

## [HSC Spectra Download]
def downloadHscSpectra():

    # grab all the hsc spectra
    request = {'service':'Mast.HscSpectra.Db.All',
               'format':'json'}   
    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    # download the first 3 spects
    for spec in outData['data'][:3]:
        # build the url
        if spec['SpectrumType'] < 2:
            dataUrl = 'https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset=' \
                      + spec['DatasetName']
        else:
            dataUrl = 'https://hla.stsci.edu/cgi-bin/ecfproxy?file_id=' \
                      + spec['DatasetName'] + '.fits'

        # build the local path
        localPath = 'hscSpectra/' + spec['DatasetName'] + ".fits"
        if not os.path.exists('hscSpectra/'):
            os.makedirs('hscSpectra/')

        # download
        urlretrieve(dataUrl, localPath)

        print(localPath)   
## [HSC Spectra Download]


## [HSC V2 Matches]
def getHSCv2Matches():
    # perform the HSC search
    result = hscV2ConeSearch()
    data = result['data']

    # get the match id
    matchId = data[0]['MatchID']

    # get detailed results for chosen match id
    request = {'service':'Mast.HscMatches.Db.v2',
               'params':{'input':matchId},
               'format':'json',
               'page':1,
               'pagesize':4}   

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [HSC V2 Matches]

## [HSC V3 Matches]
def getHSCv3Matches():
    # perform the HSC search
    result = hscV3ConeSearch()
    data = result['data']

    # get the match id
    matchId = data[0]['MatchID']

    # get detailed results for chosen match id
    request = {'service':'Mast.HscMatches.Db.v3',
               'params':{'input':matchId},
               'format':'json',
               'page':1,
               'pagesize':4}   

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [HSC V3 Matches]


## [Get VO Data]
def getVoData():

    # perform vo cone search
    voData = voConeSearch()
    voJson = voData[1]

    #colnames = [x['name'] for x in voJson[1]['fields']]
    #urlidx = colnames.index('tableURL')
    
    row = voJson['data'][2]  
    url = row['tableURL']
        
    request = {'service':'Vo.Generic.Table',
               'params':{'url':url},
               'format':'json',
               'page':1,
               'pagesize':1000}   

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [Get VO Data]


## [CAOM Crossmatch]
def crossMatchFromConeSearch():
    
    # This is a json object
    crossmatchInput = caomConeSearch()
    
    request =  {"service":"Mast.Caom.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"s_ra",
                    "decColumn":"s_dec",
                    "radius":0.001
                },
                "pagesize":1000,
                "page":1,
                "format":"json",
                "removecache":True}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData

def crossMatchFromMinimalJson():
    
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":210.8,"dec":54.3}]}
    
    request =  {"service":"Mast.Caom.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.001
                },
                "pagesize":1000,
                "page":1,
                "format":"json",
                "clearcache":True}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [CAOM Crossmatch]

## [Galex Crossmatch]
def galexCrossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"s_ra","type":"float"},
                                 {"name":"s_dec","type":"float"}],
                       "data":[{"s_ra":210.8,"s_dec":54.3}]}
    
    request =  {"service":"Mast.Galex.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"s_ra",
                    "decColumn":"s_dec",
                    "radius":0.01
                },
                "pagesize":1000,
                "page":1,
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [Galex Crossmatch]    

## [sdss Crossmatch]
def sdssCrossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":337.10977,"dec":30.30261}]} 
    
    request ={"service":"Mast.Sdss.Crossmatch",
              "data":crossmatchInput,
              "params": {
                  "raColumn":"ra",
                  "decColumn":"dec",
                  "radius":0.01
              },
              "format":"json",
              "pagesize":1000,
              "page":1}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [sdss Crossmatch]


## [2mass Crossmatch]
def twoMassCrossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":210.88447,"dec":54.332}]}
    
    request =  {"service":"Mast.2Mass.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.04
                },
                "pagesize":1000,
                "page":1,
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [2mass Crossmatch]


## [hsc2 Crossmatch]
def hscMagAper2Crossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":210.8,"dec":54.3}]}
    
    request =  {"service":"Mast.Hsc.Crossmatch.MagAper2v3",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.001
                },
                "pagesize":1000,
                "page":1,
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [hsc2 Crossmatch]

## [hscauto Crossmatch]
def hscMagAutoCrossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":210.8,"dec":54.3}]}
    
    request =  {"service":"Mast.Hsc.Crossmatch.MagAutov3",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.001
                },
                "pagesize":1000,
                "page":1,
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [hscauto Crossmatch]

## [gaia DR1 Crossmatch]
def gaiaDR1Crossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":210.8,"dec":54.3}]}
    
    request =  {"service":"Mast.GaiaDR1.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.1
                },
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [gaia DR1 Crossmatch]

## [gaia DR2 Crossmatch]
def gaiaDR2Crossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":210.8,"dec":54.3}]}
    
    request =  {"service":"Mast.GaiaDR2.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.1
                },
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [gaia DR2 Crossmatch]

## [tgas Crossmatch]
def tgasCrossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":211.09,"dec":54.3228}]}
    
    request =  {"service":"Mast.Tgas.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.2
                },
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [tgas Crossmatch]

## [tic Crossmatch]
def ticCrossmatch():

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":211.09,"dec":54.3228}]}

    request =  {"service":"Mast.Tic.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":0.2
                },
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [tic Crossmatch]

## [tic single object crossmatch]
def tic_single_object_crossmatch(ra, dec, radius):
    '''
    ra, dec, radius: all in decimal degrees
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
                "format":"json"}

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    #import IPython; IPython.embed()

    return outData
## [tic single object crossmatch]


## [Product Query]
def getCaomProducts():

    # perform the CAOM search
    result = caomConeSearch()
    data = result['data']

    # get the product group id (obsid)
    obsid = data[1]['obsid']

    # get detailed results for chosen match id
    request = {'service':'Mast.Caom.Products',
               'params':{'obsid':obsid},
               'format':'json',
               'pagesize':4,
               'page':1}   

    headers,outString = mastQuery(request)

    outData = json.loads(outString)

    return outData
## [Product Query]


## [Download Product]
def downloadOneProduct():

    # get data products
    result = getCaomProducts()
    data = result['data']

    # collect the parameters you need
    urlList = data[0]['dataURI']
    descriptionList = list(data[0]['description'])
    productTypeList = list(data[0]['dataproduct_type'])

    # Decide on filename for the download and make the product path
    filename = "mastData"
    pathList = "mastFiles/"+data[0]['obs_collection'] + '/' + data[0]['obs_id'] + '/' + data[0]['productFilename']
  
    # make the bundler request
    request = {"service":"Mast.Bundle.Request",
               "timeout":"3",
               "params":{
                   "urlList":urlList,
                   "filename":filename,
                   "pathList":pathList,
                   "descriptionList":descriptionList,
                   "productTypeList":productTypeList,
                   "extension":"curl"},
               "format":"json"}  

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    # get the download file
    downloadFile = downloadRequest(outData['url'])
    
    return downloadFile,outData['manifestUrl']


def downloadMultipleProducts():

    # get data products
    result = getCaomProducts()
    data = result['data']

    # collect the parameters you need
    urlList = [x['dataURI'] for x in data]
    descriptionList = [x['description'] for x in data]
    productTypeList = [x['dataproduct_type'] for x in data]

    # Decide on filename and paths
    filename = "mastMultiData"
    pathList = ["mastFiles/"+x['obs_collection']+'/'+x['obs_id']+'/'+x['productFilename'] for x in data]
  
    # make the bundler request
    request = {"service":"Mast.Bundle.Request",
               "timeout":"3",
               "params":{
                   "urlList":",".join(urlList),
                   "filename":filename,
                   "pathList":",".join(pathList),
                   "descriptionList":descriptionList,
                   "productTypeList":productTypeList,
                   "extension":"tar.gz"},
               "format":"json"}  

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    # get the download file
    downloadFile = downloadRequest(outData['url'])

    return downloadFile,outData['manifestUrl']
## [Download Product]


## [Direct Download]
def directDownload():

    # collect the data products
    result = getCaomProducts()
    data = result['data']

    # setting up the https connection
    server='mast.stsci.edu'
    conn = httplib.HTTPSConnection(server)

    # dowload the first products
    for i in range(len(data)):

        # make file path
        outPath = "mastFiles/"+data[i]['obs_collection']+'/'+data[i]['obs_id']
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        outPath += '/'+data[i]['productFilename']
        
        # Download the data
        uri = data[i]['dataURI']
        conn.request("GET", "/api/v0/download/file?uri="+uri)
        resp = conn.getresponse()
        fileContent = resp.read()
    
        # save to file
        with open(outPath,'wb') as FLE:
            FLE.write(fileContent)
        
        # check for file 
        if not os.path.isfile(outPath):
            print("ERROR: " + outPath + " failed to download.")
        else:
            print("COMPLETE: ", outPath)

    conn.close()
## [Direct Download]

