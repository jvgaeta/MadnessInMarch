from __future__ import division  # floating point division
import numpy as np
import math
from sklearn.metrics import log_loss
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from scipy.optimize import fmin,fmin_bfgs
import random
import urllib2

# compute log loss
def geterror(ytest, predictions):
    return log_loss(ytest, predictions)

# kenpom scraper
def import_raw_year(year):
    base_url = 'http://kenpom.com/index.php'
    url_year = lambda x: '%s?y=%s' % (base_url, str(x) if x != 2016 else base_url)
    f = requests.get(url_year(year))
    soup = BeautifulSoup(f.text, "lxml")
    table_html = soup.find_all('table', {'id': 'ratings-table'})
    thead = table_html[0].find_all('thead')
    table = table_html[0]

    for x in thead:
        table = str(table).replace(str(x), '')

    df = pd.read_html(table)[0]
    df['year'] = year
    return df
    
# scrape the scoring margins
def import_margin():
    data = None
    for year in range(2002, 2017):
        url = 'https://www.teamrankings.com/ncaa-basketball/stat/average-scoring-margin?date=' + str(year) + '-04'
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html, "lxml")
        table_html = soup.find_all('table')
        thead = table_html[0].find_all('thead')
        table = table_html[0]

        for x in thead:
            table = str(table).replace(str(x), '')

        df = pd.read_html(table)[0]
        df['Season'] = year
        df.columns = ['Rank', 'Team', 'Scoring Margin', 'Last 3' ,'Last 1', 'Home', 'Away', 'Previous Year', 'Season']
        if data is None:
            data = df
        else:
            data = pd.concat([data, df], axis=0)
    data.to_csv('ScoringMargin.csv')

# scrape the rebounding stats
def import_rebounding():
    data = None
    for year in range(2002, 2017):
        url = 'https://www.teamrankings.com/ncaa-basketball/stat/total-rebounds-per-game?date=' + str(year) + '-04'
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html, "lxml")
        table_html = soup.find_all('table')
        thead = table_html[0].find_all('thead')
        table = table_html[0]

        for x in thead:
            table = str(table).replace(str(x), '')

        df = pd.read_html(table)[0]
        df['Season'] = year
        df.columns = ['Rank', 'Team', 'RPG', 'Last 3' ,'Last 1', 'Home', 'Away', 'Previous Year', 'Season']
        if data is None:
            data = df
        else:
            data = pd.concat([data, df], axis=0)
    data.to_csv('ReboundingStats.csv')

def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def calculateprob(x, mean, stdev):
    if stdev < 1e-3:
        if math.fabs(x-mean) < 1e-2:
            return 1.0
        else:
            return 0
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    
def sigmoid(xvec):
    xvec[xvec < -100] = -100
   
    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))
 
    return vecsig

def single_val_sigmoid(value):
    if value < -100:
        value = -100

    return 1.0 / (1.0 + np.exp(value * -1))

def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)


def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
            
def fmin_simple(loss, grad, initparams):
    return fmin_bfgs(loss,initparams,fprime=grad)                

def logsumexp(a):

    awithzero = np.hstack((a, np.zeros((len(a),1))))
    maxvals = np.amax(awithzero, axis=1)
    aminusmax = np.exp((awithzero.transpose() - maxvals).transpose())

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(aminusmax, axis=1))

    out = np.add(out,maxvals)

    return out