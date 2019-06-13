# =============================================================================
# Trying to set up the "set" function so that it handles everything
# and the user just needs to fit it.
# =============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split

class CategoricalNaiveBayes:
    def __init__(self, lap_corr = 0.01, threshold = 1):
        """Initialization of Categorical Naive Bayes"""
        self.lap_corr = lap_corr
        self.list_of_probs = ''
        self.target_values = ''
        self.features = ''
        self.X_train = ''
        self.y_train = ''
        self.y_test = ''
        self.y_pred = ''
        #self.target_threshold = threshold
        #self.threshold_category = 'True'
     
    def __get_feature_categories(self, dataset, features):
        """Grabs and stores all features from a dataset """
        cat_dict = {}
        for feature in features:
            u_categories = list(dataset[feature].value_counts().index)
            u_categories.sort()
            cat_dict[feature] = u_categories
        return cat_dict  

    def __create_prob_df(self,features,feature_categories, target_values):
        """Creates the probability table that stores the final probabilites 
        used for classification"""
        prob_dict = {}
        for feature in features:
            temp_df = pd.DataFrame(0, index = feature_categories[feature], 
                                   columns = target_values)
            prob_dict[feature] = temp_df
        return prob_dict
      
    def __get_count_of_targets_from_unique_set(self, 
                                               dataset, 
                                               features, 
                                               category, 
                                               target_values):
        """Counts how many tuples are classified as target class"""
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
        """Used for when a target is not included in the training 
        set and adds it to the unique count"""
        targets_hit = list(temp.index)
        add_targets = list(target_values)
        for target in targets_hit:
            add_targets.remove(target)
        for target in add_targets:
            temp[target] = 0
        return temp
    
    def __get_probabilities(self,feat_cat_target_counts, 
                            target_categories, 
                            features, 
                            feature_categories):
        """Gets the probabilities of each category and stores them"""
        for feature in features:
            for category in feature_categories[feature]:
                for i in range(len(target_categories)):
                    self.list_of_probs[feature].\
                    loc[category, target_categories.index[i]] =\
                    feat_cat_target_counts[feature][category]\
                    [target_categories.index[i]] /\
                    target_categories[i] + self.lap_corr
                    
# =============================================================================
#     def set_data(self, X, y, test_size = 0.25):
#          self.X_train, self.X_test, self.y_train, self.y_test = \
#             train_test_split(X, y, test_size = test_size)
#         self.X_train['target'] = y_train
#         self.X_train.reset_index(drop = True, inplace = True)
#         self.y_train.reset_index(drop = True, inplace = True)
#         self.y_test.reset_index(drop = True, inplace = True)
#         self.X_test.reset_index(drop = True, inplace = True)
# =============================================================================

        
    
    def fit(self, dataset):
        """Method that trains the classifier"""
        self.features = list(dataset.iloc[:,:-1].columns) 
        target_categories = dataset.iloc[:,-1].value_counts()
        target_categories = target_categories.sort_index()
        self.target_values = list(dataset.iloc[:,-1].value_counts().index)
        self.target_values.sort()
        feature_categories = \
        self.__get_feature_categories(dataset, self.features)
        self.list_of_probs = \
        self.__create_prob_df(self.features, feature_categories, self.target_values)
        feat_cat_target_counts = \
        self.__get_count_of_targets_from_unique_set(dataset, self.features, feature_categories,self.target_values)
        self.__get_probabilities(feat_cat_target_counts, target_categories, self.features, feature_categories)

    def __set_pred_dict(self):
        """Sets up the dictionary used for predicting classes"""
        target_guess = {}
        for i in range(len(self.target_values)):
            target_guess[self.target_values[i]] = 1
        return target_guess
    
# =============================================================================
#     def __get_max_key(self, target_guess):
#         """ Gets the max from a dictionary"""
#         target_guess[self.threshold_category] = target_guess[self.threshold_category] * self.target_threshold
#         values=list(target_guess.values())
#         keys=list(target_guess.keys())
#         return keys[values.index(max(values))]
#     
# =============================================================================
    def __get_max_key(self, target_guess):
        """ Gets the max from a dictionary"""
        values=list(target_guess.values())
        keys=list(target_guess.keys())
        return keys[values.index(max(values))]
        
    def predict(self, X_test):
        """Method used to predict using the test set or new data"""
        y_pred = []
        for i in range(len(X_test)): # iterate through observations
            target_guess = self.__set_pred_dict() # Set
            for target in self.target_values: # iterate through targets
                for feature in self.features: 
                    target_guess[target] *= self.list_of_probs[feature].loc[X_test.loc[i,feature], target]
            obs = self.__get_max_key(target_guess)
            y_pred.append(obs)
        return y_pred
    

