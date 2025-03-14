# KATIE HIPPE
# AMATH 482
# 25 February 2025
# HOMEWORK 3


# start off with imports as always 
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

# hw helper code to read in the data 
with open('train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    print(data.shape)
    Xtraindata = np.transpose(data.reshape((size, nrows*ncols)))

with open('train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytrainlabels = data.reshape((size,)) # (Optional)

with open('t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtestdata = np.transpose(data.reshape((size, nrows*ncols)))

with open('t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytestlabels = data.reshape((size,)) # (Optional)


print(Xtraindata.shape)
print(ytrainlabels.shape)
print(Xtestdata.shape)
print(ytestlabels.shape)


def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
    for i in range(N):
      for j in range(N):
        #print(XX[(N)*i+j,:].shape)
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

    plt.show()

plot_digits(Xtraindata, 4, "First 16 Training Images" )




### task 1: reshape and plot first 16 PC modes

Xtraindata = Xtraindata.T
Xtestdata = Xtestdata.T


# perform analysis for the first 16 components 
pca = PCA(n_components = 16)
pca_16 = pca.fit(Xtraindata)
xnew = pca_16.transform(Xtraindata)

plot_digits(pca_16.components_.T, 4, "First 16 PC Modes")





### task 2 


# find fro norm

pca = PCA().fit(Xtraindata)

energy = pca.explained_variance_ratio_.cumsum()
n = np.where(energy > 0.85)[0][0] + 1

print('for 85 percent, use:', n)

x_axis = np.arange(1,101)
plt.plot(x_axis, energy[:100])
plt.axhline(y=.85, color='r')
plt.xlabel('Number of PC Modes', fontsize=22)
plt.ylabel('Cumulative Energy', fontsize=22)
plt.show()



# plot 16 instances of the data to see that it works out 
pca = PCA(n_components=n, random_state=42) # using our previously calculated
pca.fit(Xtraindata)
vals_pca = pca.transform(Xtraindata)
recon = pca.inverse_transform(vals_pca)

test_pca = pca.transform(Xtestdata)

plot_digits(recon.T, 4, "Image Reconstruction for k-PC Modes")



### task 3: select two specific classifications 

def subset_digits(XX, XL, XT, XTL, dig1, dig2): 

    print(XL.shape)

    # make a set of indices from the label set
    itrain = np.where((XL == dig1) | (XL == dig2))[0]
    itest = np.where((XTL == dig1) | (XTL == dig2))[0]

    print(itrain.shape)

    # select those indices from the Xtraining set
    XX = XX[itrain,:]
    XL = XL[itrain]

    XT = XT[itest,:]
    XTL = XTL[itest]

    return XX, XL, XT, XTL






### task 4: isolate digits 1 and 8 and continue analysis with just them


def analyze_pairs(dig1, dig2): 

    print("\nAnalysis for: ", dig1, ", ", dig2)

    Xtr, ytr, Xte, yte = subset_digits(Xtraindata, 
                                    ytrainlabels, Xtestdata, ytestlabels, dig1, dig2)

    

    Xtr_pca = pca.transform(Xtr) # we already fit our data using all of it 
    Xte_pca = pca.transform(Xte)

    # do the classifier!
    RidgeCL = RidgeClassifierCV()

    #fit the model!
    RidgeCL.fit(Xtr_pca, ytr)

    # evaluate how well it's doing 
    print("Training Score: {}".format(RidgeCL.score(Xtr_pca, ytr)))
    print("Testing Score: {}".format(RidgeCL.score(Xte_pca, yte)))

    # perform cross validation as well
    scores = cross_val_score(RidgeCL, Xtr_pca, ytr, cv=5)
    print("{} accuracy with a standard deviation of {}".format(scores.mean(), scores.std()))
    

analyze_pairs(1,8)



### task 5

analyze_pairs(3,8)
analyze_pairs(2,7)



### task 6

# scale and transform our testing data (vals_pca is the training)
Xte_pca = pca.transform(Xtestdata)


# first we'll do the ridge classifier methods 
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(vals_pca, ytrainlabels)

print("Training Score: {}".format(RidgeCL.score(vals_pca, ytrainlabels)))
print("Testing Score: {}".format(RidgeCL.score(Xte_pca, ytestlabels)))

scores = cross_val_score(RidgeCL, vals_pca, ytrainlabels, cv=5)
print("{} accuracy with a standard deviation of {}".format(scores.mean(), scores.std()))


# and now the KNN


KNNCL = KNeighborsClassifier(n_neighbors=3)
KNNCL.fit(vals_pca, ytrainlabels)

print("Training Score: {}".format(KNNCL.score(vals_pca, ytrainlabels)))
print("Testing Score: {}".format(KNNCL.score(vals_pca, ytrainlabels)))

k_values = [i for i in range (1,31)]
scores = []

for k in k_values:
    KNNCLk = KNeighborsClassifier(n_neighbors=k)
    scorescv = cross_val_score(KNNCLk, vals_pca, ytrainlabels,cv=5)
    print("{} accuracy with a standard deviation of {}".format(scorescv.mean(), scorescv.std()))
    scores.append(scorescv.mean())

sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")

plt.show()



## EXTRA CREDIT: an alternate classifier! linear SVM

'''
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(vals_pca, ytrainlabels)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)
'''

clf = SVC()
clf.fit(vals_pca, ytrainlabels)

print("SVC")

print("Training Score: {}".format(clf.score(vals_pca, ytrainlabels)))
print("Testing Score: {}".format(clf.score(vals_pca, ytrainlabels)))

scores = cross_val_score(clf, vals_pca, ytrainlabels, cv=5)
print("{} accuracy with a standard deviation of {}".format(scores.mean(), scores.std()))