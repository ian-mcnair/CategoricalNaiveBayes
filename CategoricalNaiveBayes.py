import pandas as pd

class CategoricalNaiveBayes:
    import pandas as pd
    def __init__(self, lap_corr = 0.0001):
        self.lap_corr = lap_corr
        self.list_of_probs = ''
        self.target_values = ''
        self.features = ''
     
    def __get_feature_categories(self, dataset, features):
        cat_dict = {}
        for feature in features:
            u_categories = list(dataset[feature].value_counts().index)
            u_categories.sort()
            cat_dict[feature] = u_categories
        return cat_dict  

    def __create_prob_df(self,features,feature_categories, target_values):
        prob_dict = {}
        for feature in features:
            temp_df = pd.DataFrame(0, index = feature_categories[feature], columns = target_values)
            prob_dict[feature] = temp_df
        return prob_dict
      
    def __get_count_of_targets_from_unique_set(self, dataset, features, category, target_values):
        big_dict = {}
        for feature in features:
            temp_dict = {}
            for j in range(len(category[feature])):
                temp = dataset[dataset[feature] == category[feature][j]]
                temp = temp.iloc[:, -1]
                temp = temp.sort_values()
                temp = temp.value_counts()
                if len(temp) < len(target_values):
                    temp = self.__add_missing_targets(temp, target_values)
                temp_dict[category[feature][j]] = temp
            big_dict[feature] = temp_dict
        return big_dict
    
    def __add_missing_targets(self, temp, target_values):
        targets_hit = list(temp.index)
        add_targets = list(target_values)
        for target in targets_hit:
            add_targets.remove(target)
        for target in add_targets:
            temp[target] = 0
        return temp
    
    def __get_probabilities(self,feat_cat_target_counts, target_categories, features, feature_categories):
        for feature in features:
            for category in feature_categories[feature]:
                for i in range(len(target_categories)):
                    self.list_of_probs[feature].loc[category, target_categories.index[i]] = feat_cat_target_counts[feature][category][target_categories.index[i]] / target_categories[i] + self.lap_corr
    
    def fit(self, dataset):
        self.features = list(dataset.iloc[:,:-1].columns) 
        target_categories = dataset.iloc[:,-1].value_counts()
        target_categories = target_categories.sort_index()
        self.target_values = list(dataset.iloc[:,-1].value_counts().index)
        self.target_values.sort()
        feature_categories = self.__get_feature_categories(dataset, self.features)
        self.list_of_probs = self.__create_prob_df(self.features, feature_categories, self.target_values)
        feat_cat_target_counts = self.__get_count_of_targets_from_unique_set(dataset, self.features, feature_categories,self.target_values)
        self.__get_probabilities(feat_cat_target_counts, target_categories, self.features, feature_categories)

    def __set_pred_dict(self):
        target_guess = {}
        for i in range(len(self.target_values)):
            target_guess[self.target_values[i]] = 1
        return target_guess
    
    def __get_max_key(self, target_guess):
        values=list(target_guess.values())
        keys=list(target_guess.keys())
        return keys[values.index(max(values))]
    
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)): # iterate through observations
            target_guess = self.__set_pred_dict() # Set
            for target in self.target_values: # iterate through targets
                for feature in self.features: 
                    target_guess[target] *= self.list_of_probs[feature].loc[X_test.loc[i,feature], target]
                    # Index NEEDS TO BE RESET
            obs = self.__get_max_key(target_guess)
            y_pred.append(obs)
        return y_pred
    

