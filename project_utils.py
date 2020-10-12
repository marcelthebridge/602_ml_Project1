# Functions for use with Project_1 notebooks
# Imports:

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_plot(hue, data):
    for i, col in enumerate(data.columns):
        plt.figure(i)
        sns.set(rc={'figure.figsize':(20, 5)})
        ax = sns.countplot(x=data[col],palette='mako',hue=hue,data=data)

def print_results(classifier, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print('Results of {} Model: \n'.format(classifier))
    print('Accuracy of model {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
    print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
    print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))   



def visual_model(title, X, y, classifier, resolution=0.05):
    


    # setup marker generator and color map
    markers = ('x', 'o')
    colors = ('black','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    plt.figure(figsize=(15,10))
    
    #plot surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        
