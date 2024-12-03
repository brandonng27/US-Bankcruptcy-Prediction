import pandas as pd
import numpy as np
from sklearn import metrics
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def set_lookback_binary_encoding(df:pd.DataFrame, lookback = 1, drop = True)-> pd.DataFrame:
    '''
    return the dataframe with the proper status_label format,
    i.e. only status = failed at the financial year
    
    the given dataframe is being deep copied, so the original dataframe would not be edited
    '''
    sample = df.set_index(['company_name','year'])
    dff = sample['status_label'].copy()

    for t in tqdm(list(df['company_name'].unique())):
        selected = dff.loc[(t),:]
        # print(selected)
        if (selected.iloc[-1] == 'failed'): 
            # in current implementation it is ok? to have companies with only one instance, 
            # as we are predicting two year ahead bankruptcy for current data
            # assuming that t-2 and t-1 data are both iid in terms of bankruptcy joint distribution
            # see if we will change this later
            if(len(selected)<lookback and not drop):
                print(dff.loc[(t)])
                dff.loc[(t)] = 1                
            # else:
            #     selected[:-lookback] = 0
            #     selected.iloc[-1] = 1
            #     dff.loc[(t)] = selected.values
            selected[:-lookback] = 0
            selected.iloc[-lookback:] = 1
            dff.loc[(t)] = selected.values
        else:
            dff.loc[(t)]  = 0
    sample['status_label'] = dff.values
    if (drop):
        sample = sample.dropna()
    return sample

# create the range of hyperparameters to search for
def model_configs(n_comps):
    # create configs
    configs = list()
    for k in n_comps:
        cfg = k
        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

class walk_forward_rolling_train:
    '''
    Inspired by ECON4305 24 Fall ASM 1 and chapter 3 code examples,
    X and Y are training (and validation) sets! do train-test split before
    
    metric: default false positive, but can specific any user specific loss function / evaluation metric
    
    train_pct and valid_pct should sum up to 1

    resampling: resample to restore balance for sample class distribution, independently for training and testing set 
    
    flip: anomaly detection models returns 1 for inliers and -1 for outliers, so we need to flip these
    '''
    def __init__(self, pipeline, X, Y, train_year = 2, valid_year = 1, metric = metrics.recall_score, squared = False, smote = SMOTE(random_state=42),flip = False):
        self.pipeline = pipeline 
        self.X = X
        self.Y = Y
        self.n_train = train_year
        self.n_valid = valid_year
        self.metric = metric
        self.squared = squared
        self.sm = smote
        self.flip = flip
    
    def walk_forward_validation(self,cfg):
        n_trains = self.n_train + self.X.index.min()
        n_records = self.X.index.max() - self.n_valid
        score_list = []
        j = self.X.index.min()
        for i in range(n_trains, n_records):
            if self.sm is not None:
                X_train, y_train = self.sm.fit_resample(self.X.loc[j:i].to_numpy(dtype='int'), self.Y.loc[j:i].to_numpy(dtype='int'))
                X_test, y_test = self.sm.fit_resample(self.X.loc[i+self.n_valid].to_numpy(dtype='int'), self.Y.loc[i+self.n_valid].to_numpy(dtype='int'))
            else:
                X_train, X_test, y_train, y_test = self.X.loc[j:i].values, self.X.loc[i+self.n_valid].values, self.Y.loc[j:i].to_numpy(dtype='int'), self.Y.loc[i+self.n_valid].to_numpy(dtype='int')
            model = self.pipeline(cfg).fit(X_train, y_train)        
            y_pred = model.predict(X_test)
            if (self.flip): 
                y_pred = np.where(y_pred == 1, 0, 1)
            metric = self.metric(y_test,y_pred)
            if (self.squared):
                metric**2
            score_list.append(metric) 
            j += 1
        print(' > %.3f' % np.mean(score_list))
        return score_list 

    def repeat_evaluate(self,config, n_repeats = 1):
        key = str(config)
        score_list = [self.walk_forward_validation(config) for _ in range(n_repeats)]
        result = np.mean(score_list[0])
        result_std = np.std(score_list[0]) # might be wrong
        print(key, result,result_std)
        return {'ModelParam':key, 'Results':(result,result_std)}
 
    def grid_search(self,cfg_list):
        self.scores = [self.repeat_evaluate(cfg) for cfg in tqdm(cfg_list)]
        self.scores.sort(key=lambda tup: tup['Results'][0], reverse=True)
        print('done')

    def plot_validation(self,standard_error = True):

        model_params = [eval(item['ModelParam']) for item in self.scores] 
        results_1 = [item['Results'][0] for item in self.scores]         
        x = [param[0] for param in model_params]  
        y = [param[1] for param in model_params]  
        X_unique = np.unique(x)
        Y_unique = np.unique(y)
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = np.full(X.shape, np.nan)  
        for i in range(len(x)):
            x_idx = np.where(X_unique == x[i])[0][0]
            y_idx = np.where(Y_unique == y[i])[0][0]
            Z[y_idx, x_idx] = results_1[i]
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X_unique, y=Y_unique, colorscale='Viridis', opacity=0.7))
        if standard_error:
            errors = [item['Results'][1] for item in self.scores] 
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=results_1,
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Data Points',
                error_z=dict(type='data', array=errors, visible=True)  # Add error bars to the z-axis
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=results_1,
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Data Points'
            ))
        fig.update_layout(
            scene=dict(
                xaxis_title='X (ModelParam[0])',
                yaxis_title='Y (ModelParam[1])',
                zaxis_title='Results[0]',
                aspectmode='cube'
            ),
            title='Cross-Validation Error Plot across Grid Search'
        )
        fig.show()