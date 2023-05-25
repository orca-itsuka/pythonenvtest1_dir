#%load_ext autoreload
#%autoreload 2

import seaborn as sns
import pandas as pd
from iris import AnalyzeIris

pd.set_option("display.max_rows", None)

iris = AnalyzeIris()
iris.get()
#iris.pair_plot(diag_kind='kde')
iris.all_supervised(n_neighbors=4)
iris.get_supervised()
df_score = iris.get_supervised().describe()
print(df_score)
best_method, best_score = iris.best_supervised()
print("BestMethod is ", best_method, " : ", "{0:0.4f}".format(best_score))
#iris.plot_feature_importances_all()
iris.visualize_decision_tree()
