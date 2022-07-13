# Khushi Baghel, B20249, 7417130808

#importing libraries
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the file
df = pd.read_csv('pima-indians-diabetes.csv')

# Question 1 
df1 = df.drop(columns = ['class']) # Considering only first 8 attributes
print(" --- Question 1 --- ")

for i in df1.columns:
    q1 = df1[i].quantile([0.25][0])
    q2 = df1[i].quantile([0.5][0])
    q3 = df1[i].quantile([0.75][0])

    iqr = q3 - q1 # IQR Inter quartile range 
    lower = q1 - 1.5*iqr # Upper bound 
    upper = q3 + 1.5*iqr # Lower bound
    
    for j in df1[i]:
        if lower > j or j > upper:
            df1.loc[df1[i] == j, i] = q2

# Question 1 A
print(" --- Question 1 A ---- ")
# Finding series with min and max values
df1_min = df1.min()
df1_max = df1.max()
# Printing the minimum values 
print("Minimum values before Normalising are: ")
print(df1_min, '\n')
print("Maximum values before Normaising are: ")
print(df1_max, '\n')

# Normalising the data b/w 5 and 12
diff = df1_max - df1_min
df1a = ((df1 - df1_min) / diff)*7 + 5

# Now the min and max are 5 and 12, data normalised b/w 5 to 12
print("Minimum values after Normalising are: ")
print(df1a.min(), '\n')
print("Maximum values after Normaising are: ")
print(df1a.max(), '\n')
    
# Question 1 B
print(" --- Question 1 B ---- ")
df1_mean = df1.mean()
df1_std = df1.std()

print("Mean Values before Standardizing are: ")
print(df1_mean, '\n')
print("Standard Deviation values before Standardizing are: ")
print(df1_std, '\n')

# Now standardising the data by subtracting mean and diving by std
df1b = (df1 - df1_mean) / df1_std
print("Mean values After Standardizing are: ")
print(df1b.mean(), '\n')
print("Standard Deviation values After Standardizing are: ")
print(df1b.std(), '\n')

# Question 2
import numpy as np
mean = np.array([0, 0]).transpose()
cov_matrix = [[13, -3], [-3, 5]]

# Generating samples
samples = np.random.multivariate_normal(mean, cov_matrix, 1000)
data = pd.DataFrame(samples, columns=['A', 'B'])

plt.scatter(data['A'], data['B'], marker='.')
plt.title('Scatter Plot')
plt.ylabel('Y - Axis')
plt.xlabel('X - Axis')
plt.show()

eigen_vals, eigen_vector = np.linalg.eig(cov_matrix)
mean_A, mean_B = data.mean()
print("The Eigen Values are: ", eigen_vals)

vec1, vec2 = eigen_vector[:, 0], eigen_vector[:, 1]
print("The first Eigen Vector: ", vec1)
print("The second Eigen Vector is: ", vec2)

print("Arrows marked graphs is shown")
plt.scatter(data['A'], data['B'], marker='.')

plt.quiver(mean_A, mean_B, vec1[0], vec1[1], angles="xy", scale=5, color='r', label="vector1")
plt.quiver(mean_A, mean_B, vec2[0], vec2[1], angles="xy", scale=5, color='r', label="vector2")
plt.title('Graph with Vector Lines')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.show()

# Q2(c)
print(" --- Question 2 C --- ")
comp1 = np.matmul(samples, vec1.T)
comp2 = np.matmul(samples, vec2.T)

pro1 = [[], []]
pro2 = [[], []]
# These components are of size 1000
for i in range(comp1.size):
    s = comp1[i]*vec1.T
    t = comp2[i]*vec2.T

    pro1[0].append(s[0])
    pro1[1].append(s[1])
    pro2[0].append(t[0])
    pro2[1].append(t[1])

# Now we will plot the graphs
print("Points along the First Vector")
plt.scatter(data['A'], data['B'], marker='.')
plt.scatter(pro1[0], pro1[1], marker='.')
plt.quiver(mean_A, mean_B, vec1[0], vec1[1], angles="xy",scale=5, color='r', label="vector1")
plt.quiver(mean_A, mean_B, vec2[0], vec2[1], angles="xy",scale=5, color='r', label="vector2")
plt.title('Graph with Projection along 1st Vector')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.show()
print()

print("Points along the second Vector")
plt.scatter(data['A'], data['B'], marker='.')
plt.scatter(pro2[0], pro2[1], marker='.')

plt.quiver(mean_A, mean_B, vec1[0], vec1[1], angles="xy", scale=5, color='r', label="vector1")
plt.quiver(mean_A, mean_B, vec2[0], vec2[1], angles="xy", scale=5, color='r', label="vector2")
plt.title('Graph with Projection along 2nd Vector')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.show()
print()

# Question 2 D
print(" --- Question 2 D ---- ")

newdf = pd.DataFrame(np.random.randn(1000, 2), columns=['A', 'B'])
# Reconstruct the data samples using all Eigen Vectors
for i in range(1000):
    newdf.iloc[i] = comp1[i] * vec1 + comp2[i] * vec2

loss = (np.sum((newdf - data) ** 2, axis = 1)** 0.5).mean()
print("The Reconstruction Error is: ", loss)
print("The Reconstruction Error (rounded to 3) is: ", round(loss, 2))


# Question 3
# A
print(" --- Question 3 A ---- ")
cov_matrix = np.cov(df1b.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
print('Two largest eigen values are: ',eigen_values[0:2])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df1b)
pcaDf = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
print('Variance of Projected data: ',pca.explained_variance_)

plt.scatter(pcaDf['principal component 1'],pcaDf['principal component 2'], marker = '.')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('Reduced dimensional data')
plt.show()

# B
print(" --- Question 3 B --- ")
e_values = np.sort(eigen_values)[::-1]
print('Eigen values in descending order: ',e_values)
elist = list(e_values)
# Sorting and plotting
elist.sort(reverse=True)
x = [i for i in range(len(elist))]
plt.bar(x, elist)
plt.title('Eigen Values sorted in Descending Order')
plt.xlabel('Position')
plt.ylabel('Eigen Value')
plt.show()

# C
print(" --- Question 3 C --- ")
error_record = []
# creating the error_record
for i in range(1, 9):
    pca = PCA(n_components=i)
    pca2_results = pca.fit_transform(df1b)
    pca2_proj_back = pca.inverse_transform(pca2_results)
    total_loss = (np.sum((pca2_proj_back - df1b) ** 2, axis = 1)** 0.5).mean()
    error_record.append(total_loss)
    
for i in range(2, 9):
    pca = PCA(n_components=i)
    red = pca.fit_transform(df1b).T
    corr_matrix = np.cov(red)
    print('Covariance matrix for l=', i)
    print(corr_matrix.round(3))

print(error_record)
x = [i for i in range(len(error_record))]
plt.plot(x, error_record)
plt.title('Reconstructive error Vs dimensions')
plt.xlabel('l dimensions')
plt.ylabel('Reconstructive Error')
plt.show()

# D
print(" --- Question 3 D --- ")
cov_matrix1 = np.cov(df1b.T)
print(cov_matrix1.round(3))
pca = PCA(n_components=8)
pca3 = pca.fit_transform(df1b)
cov_matrix2 = np.cov(pca3.T)
print(cov_matrix2.round(3))