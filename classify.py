import numpy as np 
import pandas as pd
import numpy as np
import math
import itertools 
from subprocess import check_output
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import utilities as utils
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def loaddata():
      # ken pom has a seed on there but not sure how to use it             
      valid_seed = lambda x: True if str(x).replace(' ', '').isdigit() and int(x) > 0 and int(x) <= 16 else False
      Teams = pd.read_csv('input/TeamSpellings.csv')
      submission = pd.read_csv('input/SampleSubmission2.csv')
      submission = pd.concat([submission['Id'],submission['Id'].str.split('_', expand=True)], axis=1)
      submission.rename(columns={0: 'Season', 1: 'Team1',2: 'Team2'}, inplace=True)
      submission['Season'] = pd.to_numeric(submission['Season'])
      submission['Team1'] = pd.to_numeric(submission['Team1'])
      submission['Team2'] = pd.to_numeric(submission['Team2'])
      season_results = pd.read_csv('input/RegularSeasonCompactResults.csv')
      # team dictionary so that ken pom and kaggle data will match up
      team_dict = dict(zip(Teams['name_spelling'].values, Teams['team_id'].values))
      season_results = season_results.drop(["Numot", "Wscore", "Lscore", "Daynum"], axis=1)
      season_results['Team1'] = season_results['Wteam'].copy()
      season_results['Team2'] = season_results['Lteam'].copy()
      season_results.drop(['Lteam'], inplace=True, axis=1)
      season_results = season_results.drop_duplicates(subset=['Team1', 'Team2', 'Wteam'])

      Experience = pd.read_csv('input/height.csv')
      Experience = Experience[['Season', 'TeamName', 'Size', 'Exp', 'Bench']]
      Experience.columns = ['Season', 'Team_1', 'Size', 'Experience', 'Bench']

      Experience2 = Experience.copy()
      Experience2.columns = ['Season', 'Team_2', 'Size2', 'Experience2', 'Bench2']

      # get the teams
      Experience['Team_1'] = Experience['Team_1'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)
      Experience['Team_1'] = Experience['Team_1'].map(lambda x : x.lower())
      Experience['Team_1'] = Experience['Team_1'].map(lambda x : x.strip())

      Experience2['Team_2'] = Experience2['Team_2'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)
      Experience2['Team_2'] = Experience2['Team_2'].map(lambda x : x.lower())
      Experience2['Team_2'] = Experience2['Team_2'].map(lambda x : x.strip())

      Experience['Team1'] = Experience['Team_1'].map(team_dict)
      Experience2['Team2'] = Experience2['Team_2'].map(team_dict)

      Experience.drop(['Team_1'], axis=1, inplace=True)
      Experience2.drop(['Team_2'], axis=1, inplace=True)

      Experience = Experience[pd.notnull(Experience['Team1'])]
      Experience2 = Experience2[pd.notnull(Experience2['Team2'])]

      Experience = Experience[['Season', 'Team1', 'Experience', 'Bench']]
      Experience2 = Experience2[['Season', 'Team2', 'Experience2', 'Bench2']]
      Experience['Team1'] = pd.to_numeric(Experience['Team1'])
      Experience2['Team2'] = pd.to_numeric(Experience2['Team2'])
      Experience['Season'] = pd.to_numeric(Experience['Season'])
      Experience2['Season'] = pd.to_numeric(Experience2['Season'])
      
      for i in season_results.index.get_values():
            if season_results.ix[i, 'Team1'] > season_results.ix[i, 'Team2']:
                  temp = season_results.ix[i, 'Team1']
                  season_results.ix[i, 'Team1'] = season_results.ix[i, 'Team2']
                  season_results.ix[i, 'Team2'] = temp

      season_results = season_results[season_results['Season'] >= 2002]
      season_results = season_results[season_results['Season'] <= 2011]
      base_url = 'http://kenpom.com/index.php'
      url_year = lambda x: '%s?y=%s' % (base_url, str(x) if x != 2016 else base_url)

      years = range(2002, 2016) 

      # Import all the years into a singular dataframe
      df = None
      for x in years:
            df = pd.concat((df, utils.import_raw_year(x)), axis=0) if df is not None else utils.import_raw_year(2002)

      df.columns = ['Rank', 'Team', 'Conference', 'W-L', 'Pyth', 
                    'AdjustO', 'AdjustO Rank', 'AdjustD', 'AdjustD Rank',
                    'AdjustT', 'AdjustT Rank', 'Luck', 'Luck Rank', 
                    'SOS Pyth', 'SOS Pyth Rank', 'SOS OppO', 'SOS OppO Rank',
                    'SOS OppD', 'SOS OppD Rank', 'NCSOS Pyth', 'NCSOS Pyth Rank', 'Season']

      df2 = df.copy()
      df2.columns = ['Rank2', 'Team_2', 'Conference2', 'W-L', 'Pyth2', 
                     'AdjustO2', 'AdjustO Rank2', 'AdjustD2', 'AdjustD Rank2',
                     'AdjustT2', 'AdjustT Rank2', 'Luck2', 'Luck Rank2', 
                     'SOS Pyth2', 'SOS Pyth Rank2', 'SOS OppO2', 'SOS OppO Rank2',
                     'SOS OppD2', 'SOS OppD Rank2', 'NCSOS Pyth2', 'NCSOS Pyth Rank2', 'Season']

      # get the teams
      df['Team'] = df['Team'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)
      df['Team'] = df['Team'].map(lambda x : x.lower())
      df['Team'] = df['Team'].map(lambda x : x.strip())

      df2['Team_2'] = df2['Team_2'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)
      df2['Team_2'] = df2['Team_2'].map(lambda x : x.lower())
      df2['Team_2'] = df2['Team_2'].map(lambda x : x.strip())

      df['Wins1'] = df['W-L'].map(lambda x : float(x.split('-')[0]))
      df2['Wins2'] = df2['W-L'].map(lambda x : float(x.split('-')[0]))


      df = df[[ 'Season', 'Wins1', 'Team','Pyth', 'AdjustO', 'AdjustD', 'Luck', 'SOS Pyth']]
      df2 = df2[[ 'Season', 'Wins2', 'Team_2', 'Pyth2' , 'AdjustO2','AdjustD2', 'Luck2', 'SOS Pyth2']]

      #df = df[[ 'Season', 'Team', 'AdjustO', 'AdjustD', 'Luck', 'SOS Pyth']]
      #df2 = df2[[ 'Season', 'Team_2', 'AdjustO2','AdjustD2', 'Luck2', 'SOS Pyth2']]


      df['Team1'] = df['Team'].map(team_dict)
      df2['Team2'] = df2['Team_2'].map(team_dict)
      df.drop('Team', inplace=True, axis=1)
      df2.drop('Team_2', inplace=True, axis=1)

      # some teams don't exist anymore this is okay
      df = df[pd.notnull(df['Team1'])]
      df2 = df2[pd.notnull(df2['Team2'])]
      
      # scoring_margin = scoring_margin[pd.notnull(scoring_margin['Team1'])]
      # scoring_margin2 = scoring_margin2[pd.notnull(scoring_margin2['Team2'])]
      
      # rebounding = rebounding[pd.notnull(rebounding['Team1'])]
      # rebounding2 = rebounding2[pd.notnull(rebounding2['Team2'])]
      train_data = pd.merge(season_results, df, on=['Season', 'Team1'], how='inner')
      #train_data = pd.merge(train_data, Experience, on=['Season', 'Team1'], how='inner')
      #train_data = pd.merge(train_data, scoring_margin, on=['Season', 'Team1'], how='inner')
      #train_data = pd.merge(train_data, rebounding, on=['Season', 'Team1'], how='inner')
      train_data = pd.merge(train_data, df2, on=['Season', 'Team2'], how='inner')
      #train_data = pd.merge(train_data, Experience2, on=['Season', 'Team2'], how='inner')
      #train_data = pd.merge(train_data, scoring_margin2, on=['Season', 'Team2'], how='inner')
      #train_data = pd.merge(train_data, rebounding2, on=['Season', 'Team2'], how='inner')
      ytrain = train_data['Wteam'].copy()

      for i in range(0, train_data.shape[0]):
	     if train_data.ix[i, 'Team1'] == train_data.ix[i, 'Wteam']:
		    ytrain[i] = 1
	     else:
		    ytrain[i] = 0

      Xtrain = train_data.copy()
      Xtrain = Xtrain.drop(['Team1', 'Team2', 'Wteam', 'Season', 'Wloc'], axis=1)
      Xtest = pd.merge(submission, df, on=['Season', 'Team1'], how='inner')
      #Xtest = pd.merge(Xtest, Experience, on=['Season', 'Team1'], how='inner')
      #Xtest = pd.merge(Xtest, scoring_margin, on=['Season', 'Team1'], how='inner')
      #Xtest = pd.merge(Xtest, rebounding, on=['Season', 'Team1'], how='inner')
      Xtest = pd.merge(Xtest, df2, on=['Season', 'Team2'], how='inner')
      #Xtest = pd.merge(Xtest, Experience2, on=['Season', 'Team2'], how='inner')
      #Xtest = pd.merge(Xtest, scoring_margin2, on=['Season', 'Team2'], how='inner')
      #Xtest = pd.merge(Xtest, rebounding2, on=['Season', 'Team2'], how='inner')
      Xtest.sort_values(['Season','Team1', 'Team2'], ascending=[True, True, True], inplace=True)
      Xtest = Xtest.drop(['Team1', 'Team2', 'Id', 'Season'], axis=1)
      
      return Xtrain, ytrain, Xtest, submission

if __name__ == '__main__':
      Xtrain, ytrain, Xtest, submission = loaddata()
      clf1 = GaussianNB()
      clf2 = LogisticRegression()
      #clf2 = GradientBoostingClassifier(n_estimators=1200, learning_rate=.01)
      clf3 = RandomForestClassifier(n_estimators=300)
      #clf = EnsembleVoteClassifier(clfs=[clf1, clf2], weights=[1, 1], voting='soft')
      print('Fitting Model...')
      clf3.fit(Xtrain, ytrain)
      print('Making Predictions...')    
      probs = clf3.predict_proba(Xtest)

      # submission
      prediction = np.clip(probs[:,1], 0.01, 0.99)
      submission['Pred'] = prediction
      submission = submission.drop(['Season', 'Team1', 'Team2'], axis=1)
      submission.to_csv('submission.csv', index=False)


