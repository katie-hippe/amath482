import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from scipy.stats import mode

# load in the dataaaa

# three empty vectors
jump = np.array([])
run = np.array([])
walk = np.array([])

# load in each data frame
folder = "hw2data/train/"
for i in range(5):

    # first get the jumping
    fname = "jumping_"+str(i+1)
    vals = np.load(folder+fname+".npy")
    if len(jump) == 0:
        jump = vals
    else:
        jump = np.concatenate([jump, vals], axis = 1)

     # then the running
    fname = "running_"+str(i+1)
    vals = np.load(folder+fname+".npy")
    if len(run) == 0:
        run = vals
    else:
        run = np.concatenate([run, vals], axis = 1)

    # then the walking 
    fname = "walking_"+str(i+1)
    vals = np.load(folder+fname+".npy")
    if len(walk) == 0:
        walk = vals
    else:
        walk = np.concatenate([walk, vals], axis = 1)


# then vstack walk-jump-run
vals = np.vstack((walk.T, jump.T, run.T))

vals = vals - np.mean(vals, axis=0)

# load in testing data as well (hardcoded as only 3)
walk = np.load("hw2data/test/walking_1t.npy")
jump = np.load("hw2data/test/jumping_1t.npy")
run = np.load("hw2data/test/running_1t.npy")
test_vals = np.vstack((walk.T, jump.T, run.T))

test_vals = test_vals - np.mean(test_vals, axis = 0) # center



## task 1 get the PCA stuff



# and scale it
scaler = StandardScaler()
scaler.fit(vals)
vals_scaled = scaler.transform(vals)

test_vals_scaled = scaler.fit_transform(test_vals)


## ALTERNATE METHOD: JUST CENTER DON'T SCALE
#vals_scaled = vals
#test_vals_scaled = test_vals

# use sklearn's PCA to determine 
pca = PCA()

# fit the pca for all the data to find the max frobenius norm 
vals_pca_all = pca.fit(vals_scaled)
max_norm = np.linalg.norm(vals_scaled, 'fro')

# intialize to hold onto each fro norm for our k values
fro_norms = []
components_range = range(1, vals.shape[1] + 1, 1)

# loop through the PCA components to see how the frobenius norm compares 
for i in range(vals.shape[1]):
    pca = PCA(n_components = i)
    vals_pca = pca.fit_transform(vals_scaled)
    #vals_ipca = pca.inverse_transform(vals_pca)

    fro_norm = np.linalg.norm(vals_pca, 'fro')

    fro_norms.append(fro_norm / max_norm)

    #print(i, ":", fro_norm / max_norm)


# graphing stuff 
plt.plot(components_range, fro_norms, label="Frobenius Norm")
plt.xlabel("PCA Modes", fontsize = 25)
plt.ylabel("Energy", fontsize = 25)
plt.axhline(.7, color='lightcoral', linestyle='--', label='70% Accuracy')
plt.axhline(.8, color='indianred', linestyle='--', label='80% Accuracy')
plt.axhline(.9, color='firebrick', linestyle='--', label='90% Accuracy')
plt.axhline(.95, color='maroon', linestyle='--', label='95% Accuracy')
plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.show()



# task 2 graphingggg



# truncate to 2 or 3 modes 
vals_pca2 = PCA(n_components = 2).fit_transform(vals_scaled)
vals_pca3 = PCA(n_components = 3).fit_transform(vals_scaled)

midpoint = len(vals_pca3[:,0]) // 3 # this is 1/3 of the way down to separate our big vals data
labels = ["Walking", "Jumping", "Running"]

# plot the 2D graph for each! 
for i in range(3):
    plt.plot(vals_pca2[:,0][midpoint*i:midpoint*(i + 1)], vals_pca2[:,1][midpoint*i:midpoint*(i+1)], '.',
              label = labels[i])

plt.xlabel("PC1", fontsize=20)
plt.ylabel("PC2", fontsize=20)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# plot the 3d crap 

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

for i in range(3):
    ax.plot(vals_pca3[:,0][midpoint*i:midpoint*(i + 1)], vals_pca3[:,1][midpoint*i:midpoint*(i+1)],
            vals_pca3[:,2][midpoint*i:midpoint*(i+1)], '.', label = labels[i], alpha=0.7, linewidth = 2)
    
ax.set_xlabel("PC1", fontsize=20)
ax.set_ylabel("PC2", fontsize=20)
ax.set_zlabel("PC3", fontsize=20)
plt.legend(fontsize=15)
plt.show()



## task 3 / 4 (function formation) + task 5 evaluating test data



# make our truth vector
truth_w = np.repeat(0, 500)
truth_j = np.repeat(1, 500)
truth_t = np.repeat(2, 500)

truth = np.concatenate((truth_w, truth_j, truth_t), axis = 0) # walk jump run
truth = np.array(truth)

# and testing ground truth 
test_truth = np.array(np.concatenate((truth_w[0:100], truth_j[0:100], truth_t[0:100]), axis = 0))

# define function to get centroid value
def centroid(k, vals):
    k_pca = PCA(n_components=k).fit_transform(vals)

    # take slices 
    walk = k_pca[:500,:]
    jump = k_pca[501:1000,:]
    run = k_pca[1001:,:]

    return np.mean(walk, axis=0), np.mean(jump, axis=0), np.mean(run, axis=0)

# initialize an accuracy score vector 
accuracies = []
test_accuracies = []

# intialize accuracies scores for our extra credit too
accuracies_kluster = []
test_accuracies_kluster = []

## EXTRA CREDIT: 
def map_klusters(pred, truth):
    labels = np.zeros_like(pred)
    for i in range(3):
        mask = (pred==i)
        labels[mask] = mode(truth[mask])[0]
    return labels

# decide on k
for k in range(1, 115): # start k at one for ease of my life

    # first we get our centroid for our particular k value
    w, r, j = centroid(k, vals_scaled)
    agg_centroid = np.array([w, r, j])


    # loop through each sample to find each sample in k-PCA space
    train_labels = []
    pca = PCA(n_components=k)
    pca.fit(vals_scaled) # put into k-PCA space
    k_pca = pca.transform(vals_scaled)

    test_labels = []
    test_k_pca = pca.transform(test_vals_scaled) # as well as test values




    # EXTRA: find the cluster centroids using kmeans 

    # first do it for the training data 
    kmeans = KMeans(n_clusters=3, random_state=10) # use random_state = 10 even if this isn't optimal
    y_kmeans = kmeans.fit(k_pca)
    # fix our offsets 
    kluster_labels = map_klusters(y_kmeans.labels_, truth)
    # add to accuracy count
    accuracies_kluster.append(accuracy_score(truth, kluster_labels))

    # then for the testing data 
    test_klusters = kmeans.predict(test_k_pca)
    # fix offsets
    kluster_labels = map_klusters(test_klusters, test_truth)
    # add to accuracy count
    test_accuracies_kluster.append(accuracy_score(test_truth, kluster_labels))




    ## ADVICE FROM ROHIN: JUST CENTER DON'T SCALE IT'LL IMPROVE ACCURACY 

    for j in range(len(truth)):
        
        curr_val = k_pca[j,:] # take slice 

        # compare each sample to the means 
        label = np.argmin(np.array([np.linalg.norm(agg_centroid[i] - curr_val) for i in range(3)]))

        train_labels.append(label)

    # same exact thing but for the testing data 
    for j in range(len(test_truth)):

        curr_val = test_k_pca[j,:] # take slice
        label = np.argmin(np.array([np.linalg.norm(agg_centroid[i] - curr_val) for i in range(3)]))
        test_labels.append(label)
    
    # now we may compare train_labels to our ground_truth!
    accuracy = accuracy_score(truth, train_labels)
    accuracies.append(accuracy)

    # as well as test labels!
    test_accuracies.append(accuracy_score(test_truth, test_labels))

# plot train / test accuracies as a function of k
plt.plot(np.arange(1,31), accuracies[0:30], label = 'Training Data')
plt.plot(np.arange(1,31), test_accuracies[0:30], label = 'Testing Data')
plt.legend(fontsize = 15)
plt.xlabel("k Modes", fontsize = 20)
plt.ylabel("Percent Accuracy of Classification", fontsize = 20)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

print(np.argmax(accuracies))
print(np.argmax(test_accuracies))


# ONE MORE TIME FOR k-means clustering


# and again for test accuracies
plt.plot(np.arange(1,31), accuracies_kluster[0:30], label="Training Data")
plt.plot(np.arange(1,31), test_accuracies_kluster[0:30], label="Testing Data")
plt.legend(fontsize = 15)
plt.xlabel("k Modes", fontsize = 20)
plt.ylabel("Percent Accuracy of Classification", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

print(np.argmax(accuracies_kluster))
print(np.argmax(test_accuracies_kluster))