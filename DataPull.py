import requests
import numpy as np
import pandas as pd
import quandl
import datetime
from datapackage import Package
import holidays
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io

quandl.ApiConfig.api_key = "RxidFKB69HRV8VFHbXqM"


def get_data(t):
    data = quandl.get('WIKI/'+t)
    return data


def get_price_data(t, start, end):
    start_date = format_date(start)
    end_date = format_date(end)
    data = quandl.get('Wiki/'+ t, start_date=start_date, end_date=end_date)
    return data


def format_date(date):
    formatted = str(date.year)
    if date.month < 10:
        formatted += '-0' + str(date.month) + '-'
    else:
        formatted += '-' + str(date.month) + '-'
    if date.day < 10:
        formatted += '0' + str(date.day)
    else:
        formatted += str(date.day)
    return formatted


def market_days(start, end):
    all_holidays = list((holidays.US(state='NY', years = datetime.datetime.today().year)+holidays.EuropeanCentralBank()).keys())
    indexes = [1, 4, 6, 7, 11, 12, 13, 14, 18]
    nyse_holidays = []
    for i in indexes:
        nyse_holidays.append(all_holidays[i])
    days = np.busday_count(start, end, holidays=nyse_holidays)
    return days


def check_open(date):
    all_holidays = list((holidays.US(state='NY', years=datetime.datetime.today().year) + holidays.EuropeanCentralBank()).keys())
    indexes = [1, 4, 6, 7, 11, 12, 13, 14, 18]
    for i in indexes:
        if date == all_holidays[i]:
            return False
    if date.weekday() > 4:
        return False
    return True


def get_all_tickers(market):
    if market == "SP500":
        package = Package('http://datahub.io/core/s-and-p-500-companies/datapackage.json')
        data = package.resources[0].read()
        tickers = []
        for i in range(0,len(data)):
            tickers.append(data[i][0])
        return tickers
    elif market == "Russell3000":
        return convert_pdf_to_txt("ru3000_members.pdf")


        # Code to automatically download the Russell3000 Constituent List (not currently working)
        # ------------------------------------------------------------------------------------------
        #r = requests.get("https://www.ftserussell.com/files/support-documents/2017-ru3000-membership-list", stream=True)
        #with open('/tmp/metadata.pdf', 'wb') as f:
        #    print(f.write(r.content))
        #return
    else:
        return


def not_ticker(inputString):
    return any(char.isdigit() or char.islower() or char == '&' for char in inputString)


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()

    words = text.split()
    words = [s for s in words if len(s) < 5 and not not_ticker(s)]
    words = list(set(words))
    return words
