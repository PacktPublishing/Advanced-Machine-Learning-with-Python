# Jitter test code
# script excerpted from Alexander Minushkin's excellent Python notebook available at https://www.kaggle.com/miniushkin/introducing-kaggle-scripts/jitter-test-for-overfitting-notebook/notebook


from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import sklearn
#sklearn two moons generator makes lots of these...
import warnings


def jitter(X, scale):
    #out = X.copy()
    if scale > 0:
        return X + np.random.normal(0, scale, X.shape)
    return X

def jitter_test(classifier, X, y, metric_FUNC = accuracy_score, sigmas = np.linspace(0, 0.5, 30), averaging_N = 5):
    out = []

    for s in sigmas:
        averageAccuracy = 0.0
        for x in range(averaging_N):
            averageAccuracy += metric_FUNC( y, classifier.predict(jitter(X, s)))

        out.append( averageAccuracy/averaging_N)

    return (out, sigmas, np.trapz(out, sigmas))

allJT = {}



#plotting functions and a series of models to demonstrate the performance of different model types under a variety of jitter conditions. In order:
    #low noise, plenty of samples, should be easy
    #more noise, plenty of samples
    #less noise, few samples
    #more noise, less samples, should be hard




def plotter(model, X, Y, ax, npts=5000):
    """
    Simple way to get a visualization of the decision boundary 
    by applying the model to randomly-chosen points
    could alternately use sklearn's "decision_function"
    at some point it made sense to bring pandas into this
    """
    xs = []
    ys = []
    cs = []
    for _ in range(npts):
        x0spr = max(X[:,0])-min(X[:,0])
        x1spr = max(X[:,1])-min(X[:,1])
        x = np.random.rand()*x0spr + min(X[:,0])
        y = np.random.rand()*x1spr + min(X[:,1])
        xs.append(x)
        ys.append(y)
        cs.append(model.predict([x,y]))
    ax.scatter(xs,ys,c=list(map(lambda x:'lightgrey' if x==0 else 'black', cs)), alpha=.35)
    ax.hold(True)
    ax.scatter(X[:,0],X[:,1],
                 c=list(map(lambda x:'r' if x else 'lime',Y)), 
                 linewidth=0,s=25,alpha=1)
    ax.set_xlim([min(X[:,0]), max(X[:,0])])
    ax.set_ylim


#next - creation of dummy data w/ variable sample counts and noise levels.

warnings.filterwarnings("ignore", category=DeprecationWarning)

Xs = []
ys = []

#low noise, plenty of samples, should be easy
X0, y0 = sklearn.datasets.make_moons(n_samples=1000, noise=.05)
Xs.append(X0)
ys.append(y0)

#more noise, plenty of samples
X1, y1 = sklearn.datasets.make_moons(n_samples=1000, noise=.3)
Xs.append(X1)
ys.append(y1)


X2, y2 = sklearn.datasets.make_moons(n_samples=200, noise=.05)
Xs.append(X2)
ys.append(y2)

#more noise, less samples, should be hard
X3, y3 = sklearn.datasets.make_moons(n_samples=200, noise=.3)
Xs.append(X3)
ys.append(y3)




classifier = sklearn.linear_model.LogisticRegression()

#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,13))

allJT[str(classifier)] = list()

i=0
for X,y in zip(Xs,ys): 
    classifier.fit(X,y)
    #plotter(classifier,X,y,ax=axes[i//2,i%2])
    allJT[str(classifier)].append (jitter_test(classifier, X, y))
    i += 1
#plt.show()


classifier = sklearn.tree.DecisionTreeClassifier()

allJT[str(classifier)] = list()

#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,13))
i=0
for X,y in zip(Xs,ys): 
    classifier.fit(X,y)
    #plotter(classifier,X,y,ax=axes[i//2,i%2])
    allJT[str(classifier)].append (jitter_test(classifier, X, y))
    i += 1
#plt.show()


classifier = sklearn.svm.SVC()

allJT[str(classifier)] = list()

#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,13))
i=0
for X,y in zip(Xs,ys): 
    classifier.fit(X,y)
    #plotter(classifier,X,y,ax=axes[i//2,i%2])
    allJT[str(classifier)].append (jitter_test(classifier, X, y))
    i += 1
#plt.show()




fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,10))

handlers =[]
for c_name in allJT:
    for i in range(4): 

        ax=axes[i//2,i%2]
        
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0.7, 1.1])

        accuracy, sigmas, area = allJT[c_name][i]
        ax.plot( sigmas, accuracy, label = "Area: {:.2} ".format(area) + c_name.split("(")[0])
        ax.legend()
    
plt.show()