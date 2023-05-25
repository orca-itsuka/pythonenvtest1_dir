#%load_ext autoreload
#%autoreload 2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
#pd.set_option("display.max_rows", None)

class AnalyzeIris:
    def __init__(self):
        self.df_iris = sns.load_dataset('iris')
        self.df_label = self.df_iris['species']
        self.df_data = self.df_iris.drop('species', axis=1)
        self.dict_train = {}
        self.dict_test = {}
        self.df_score_supervised = None
        self.label_list = (self.df_data.columns.values).tolist()
        self.DTC_feature_importances = []
        self.RFC_feature_importances = []
        self.GBC_feature_importances = []
        self.model_DTC = None

    def data_transform(self):
        self.df_label_encoded, self.df_categories = self.df_label.factorize()
        self.df_iris_encoded = pd.concat([self.df_data, pd.DataFrame(self.df_label_encoded,columns = ['species'])],axis=1)
        self.df_dummies = pd.get_dummies(self.df_label, columns=['species'])
        self.df_iris_OneHotencoded = pd.get_dummies(self.df_dummies)

    def get(self):
        AnalyzeIris.data_transform(self)
        print(self.df_iris_encoded)

    def pair_plot(self, diag_kind='hist', diag_kws=None):
        sns.pairplot(self.df_iris, hue='species', diag_kind=diag_kind, diag_kws=diag_kws)
        plt.show()

    def kfold(self,n_splits=5):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def logreg(self):
        LR = LogisticRegression(random_state=42, max_iter=200)
        #print(self.df_iris_OneHotencoded)
        cv_LR = cross_validate(LR, self.df_data, self.df_label_encoded, cv=self.kfold, return_train_score=True)
        self.dict_train['LogisticRegression'] = cv_LR['train_score']
        self.dict_test['LogisticRegression'] = cv_LR['test_score']

    def linsvc(self):
        Lsvc = LinearSVC(random_state=42, tol=1e-5, max_iter=10000)
        cv_Lsvc = cross_validate(Lsvc, self.df_data, self.df_label, cv=self.kfold, return_train_score=True)
        self.dict_train['LinearSVC'] = cv_Lsvc['train_score']
        self.dict_test['LinearSVC'] = cv_Lsvc['test_score']

    def svc(self):
        Svc = SVC(gamma='auto')
        cv_Svc = cross_validate(Svc, self.df_data, self.df_label, cv=self.kfold, return_train_score=True)
        self.dict_train['SVC'] = cv_Svc['train_score']
        self.dict_test['SVC'] = cv_Svc['test_score']

    def decisiontree(self):
        DTC = DecisionTreeClassifier(random_state=42)
        cv_DTC = cross_validate(DTC, self.df_data, self.df_iris_OneHotencoded, cv=self.kfold, return_train_score=True)
        self.dict_train['DecisionTreeClassifier'] = cv_DTC['train_score']
        self.dict_test['DecisionTreeClassifier'] = cv_DTC['test_score']
        self.model_DTC = DTC.fit(self.df_data, self.df_iris_OneHotencoded)
        self.DTC_feature_importances = DTC.feature_importances_.tolist()

        #self.dict_feature_importances['importance_DTC'] = DTC.feature_importances_.tolist()

    def knn(self, n_neighbors=4):
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        cv_KNN = cross_validate(KNN, self.df_data, self.df_iris_OneHotencoded, cv=self.kfold, return_train_score=True)
        self.dict_train['KNeighborsClassifier'] = cv_KNN['train_score']
        self.dict_test['KNeighborsClassifier'] = cv_KNN['test_score']

    def linreg(self):
        LinReg = LinearRegression()#.fit(self.df_data, self.df_iris_OneHotencoded)
        cv_LinReg = cross_validate(LinReg, self.df_data, self.df_label_encoded, cv=self.kfold, return_train_score=True)
        self.dict_train['LinearRegression'] = cv_LinReg['train_score']
        self.dict_test['LinearRegression'] = cv_LinReg['test_score']

    def rfc(self):
        RFC = RandomForestClassifier(random_state=42)
        cv_RFC = cross_validate(RFC, self.df_data, self.df_iris_OneHotencoded, cv=self.kfold, return_train_score=True)
        self.dict_train['RandomForestClassifier'] = cv_RFC['train_score']
        self.dict_test['RandomForestClassifier'] = cv_RFC['test_score']
        RFC.fit(self.df_data, self.df_iris_OneHotencoded)
        self.RFC_feature_importances = RFC.feature_importances_.tolist()
        #self.dict_feature_importances['importance_RFC'] = RFC.feature_importances_.tolist()

    def gbc(self):
        GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
        cv_GBC = cross_validate(GBC, self.df_data, self.df_label_encoded, cv=self.kfold, return_train_score=True)
        self.dict_train['GradientBoostingClassifier'] = cv_GBC['train_score']
        self.dict_test['GradientBoostingClassifier'] = cv_GBC['test_score']
        GBC.fit(self.df_data, self.df_label_encoded)
        self.GBC_feature_importances = GBC.feature_importances_.tolist()
        #self.dict_feature_importances['importance_GBC'] = GBC.feature_importances_.tolist()

    def mlp(self):
        MLP = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000, random_state=42)
        cv_MLP = cross_validate(MLP, self.df_data, self.df_label_encoded, cv=self.kfold, return_train_score=True)
        self.dict_train['MLPClassifier'] = cv_MLP['train_score']
        self.dict_test['MLPClassifier'] = cv_MLP['test_score']

    def all_supervised(self, LogReg=True, LinSVC=True, SVC=True, DTC=True, KNN=True, LinReg=True, RFC=True, GBC=True, MLP=True, n_neighbors=4):
        AnalyzeIris.kfold(self)
        if LogReg==True :
            AnalyzeIris.logreg(self)
            print('=== LogisticRegression ===')
            for i in range(len(self.dict_test['LogisticRegression'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['LogisticRegression'][i],self.dict_train['LogisticRegression'][i]))
            print("")

        if LinSVC==True :
            AnalyzeIris.linsvc(self)
            print('=== LinearSVC ===')
            for i in range(len(self.dict_test['LinearSVC'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['LinearSVC'][i],self.dict_train['LinearSVC'][i]))
            print("")

        if SVC==True :
            AnalyzeIris.svc(self)
            print('=== SVC ===')
            for i in range(len(self.dict_test['SVC'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['SVC'][i],self.dict_train['SVC'][i]))
            print("")

        if DTC==True :
            AnalyzeIris.decisiontree(self)
            print('=== DecisionTreeClassifier ===')
            for i in range(len(self.dict_test['DecisionTreeClassifier'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['DecisionTreeClassifier'][i],self.dict_train['DecisionTreeClassifier'][i]))
            print("")

        if KNN==True :
            AnalyzeIris.knn(self, n_neighbors)
            print('=== KNeighborsClassifier ===')
            for i in range(len(self.dict_test['KNeighborsClassifier'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['KNeighborsClassifier'][i],self.dict_train['KNeighborsClassifier'][i]))
            print("")

        if LinReg==True :
            AnalyzeIris.linreg(self)
            print('=== LinearRegression ===')
            for i in range(len(self.dict_test['LinearRegression'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['LinearRegression'][i],self.dict_train['LinearRegression'][i]))
            print("")

        if RFC==True :
            AnalyzeIris.rfc(self)
            print('=== RandomForestClassifier ===')
            for i in range(len(self.dict_test['RandomForestClassifier'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['RandomForestClassifier'][i],self.dict_train['RandomForestClassifier'][i]))
            print("")

        if GBC==True :
            AnalyzeIris.gbc(self)
            print('=== GradientBoostingClassifier ===')
            for i in range(len(self.dict_test['GradientBoostingClassifier'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['GradientBoostingClassifier'][i],self.dict_train['GradientBoostingClassifier'][i]))
            print("")

        if MLP==True :
            AnalyzeIris.mlp(self)
            print('=== MLPClassifier ===')
            for i in range(len(self.dict_test['MLPClassifier'])):
                print('test score: {:.3f}, train score: {:.3f}'.format(self.dict_test['MLPClassifier'][i],self.dict_train['MLPClassifier'][i]))
            print("")

    def get_supervised(self):
        test_score_series = {}
        for key, value in self.dict_test.items():
            test_score_series[key] = pd.Series(value)
        self.df_score_supervised = pd.DataFrame(test_score_series)
        print(self.df_score_supervised.head(len(self.df_score_supervised.index)))

        return self.df_score_supervised

    def best_supervised(self):
        df_mean_score = self.df_score_supervised.describe().iloc[1]
        best_score = df_mean_score.max()
        best_method = df_mean_score.idxmax()
        return best_method, best_score

    def plot_feature_importances(self, n_features, importances_list, feature_names):
        plt.barh(range(n_features), importances_list, align='center')
        plt.yticks(np.arange(n_features), feature_names)
        plt.xlabel('Feature_importance')
        plt.ylabel('Feature')

    def plot_feature_importances_all(self):
        feature_importances_list = [self.DTC_feature_importances, self.RFC_feature_importances, self.GBC_feature_importances]
        print(len(feature_importances_list))
        for i in range(len(feature_importances_list)):
            AnalyzeIris.plot_feature_importances(self, len(self.label_list), feature_importances_list[i], self.label_list)
            plt.show()
        #fig, axes = plt.subplots(len(feature_importances_list), 1, figsize=(16, 12))
        #for i in range(len(feature_importances_list)):
        #    axes[i][0].()

        #fig = plt.figure(figsize=(20,20))

        #for i in range()

    def visualize_decision_tree(self):
        plt.figure(figsize=(30,15))
        print(self.label_list)
        plot_tree(self.model_DTC, self.label_list, filled=True)
        plt.show()
