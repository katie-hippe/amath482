## Katie Hippe 
## AMATH 482/582
## January 26, 2025
## HOMEWORK 1
## this script analyzes noisy data from a 3D domain to reconstruct the true path of a submarine
## emitting a certain acoustic frequency


## PART 0: IMPORTS AND GETTING SET UP


import numpy as np
# import libraries for plotting isosurfaces
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# load in data (saved in the same file folder as this script)
d = np.load('subdata.npy')

# plot the data in time 

# NOTE: L we defined in class is 2Lh here, i.e. the domain here is [-Lh,Lh].
Lh = 10; # length of spatial domain (cube of side L = 2*10). 
N_grid = 64; # number of grid points/Fourier modes in each direction
xx = np.linspace(-Lh, Lh, N_grid+1) #spatial grid in x dir
x = xx[0:N_grid]
y = x # same grid in y,z direction
z = x

K_grid = (2*np.pi/(2*Lh))*np.linspace(-N_grid/2, N_grid/2 -1, N_grid) # frequency grid for one coordinate
kx, ky, kz = np.meshgrid(K_grid, K_grid, K_grid)

xv, yv, zv = np.meshgrid( x, y, z) # generate 3D meshgrid for plotting

# visualize our current noisy data disregarding time
plot_data = []
for j in range(0,49,3):

  signal = np.reshape(d[:, j], (N_grid, N_grid, N_grid)) # slice our data for this particular time step
  normal_sig_abs = np.abs(signal)/np.abs(signal).max()

  # generate data for isosurface of the 3D data 
  fig_data = go.Isosurface( x = xv.flatten(), y = yv.flatten(), z = zv.flatten(),
                           value = normal_sig_abs.flatten(), isomin=0.6, isomax=1)

  plot_data.append(fig_data)

# plot all our noisy data 
fig=go.Figure(data=plot_data)
fig.update_layout(
    title='Noisy Signal Data',
    scene=dict(
        xaxis=dict(title = 'X',nticks=5, tickfont=dict(size=15),titlefont=dict(size=30)),
        yaxis=dict(title = 'Y',nticks=5, tickfont=dict(size=15),titlefont=dict(size=30)),
        zaxis=dict(title = 'Z',tickvals= [-5,0,5,10], tickfont=dict(size=15),titlefont=dict(size=30))
    )
)
fig.show()


## PART 1: AVERAGE THE FOURIER TRANSFORM TO DETERMINE CENTER FREQUENCY


# create an empty 64x64x64 grid
fft_tot_signal = []

for j in range (0, 49): # loop across all time steps 

    signal = np.reshape(d[:, j], (N_grid, N_grid, N_grid)) # slice our data for this particular time step

    fft_signal = np.fft.fftn(signal) # take fft in 3 dimensions
    fft_signal = (1/N_grid)*np.fft.fftshift(fft_signal) # shift it to expected dimensions

    fft_tot_signal.append(fft_signal) # add these magnitudes to our grand grid

# now we find the coordinate of the highest frequency 
avg = np.mean(fft_tot_signal, axis = 0)
index_3d = np.unravel_index(np.argmax(avg), avg.shape)

# find frequency coordinate 
xcen, ycen, zcen = K_grid[index_3d[0]], K_grid[index_3d[1]], K_grid[index_3d[2]]

## visualize?
# isolate along our paths of travel
xsum = avg[:,index_3d[1],index_3d[2]]
ysum = avg[index_3d[0],:,index_3d[2]]
zsum = avg[index_3d[0],index_3d[1],:]
sums = [xsum, ysum, zsum]

xlabel = ["Kx", "Ky", "Kz"]

# plot X, Y, and Z axes
for i in range(3): 

    plt.plot(K_grid,sums[i])
    plt.xlabel(xlabel[i], fontsize = 25)
    plt.ylabel("Frequency Coefficient", fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axvline(x=K_grid[index_3d[i]], color='red', linestyle='--')
    plt.show()


## PART 2: DESIGN AND IMPLEMENT A FILTER TO DENOISE THE DATA 


# first find the variance from our signal data
var = np.var(fft_tot_signal)
tau = 1/(2*(var))

# create a 3D gaussian centered about the central frequency
gaussian = np.exp(-tau * ((kx - xcen)**2 + 
                  (ky - ycen)**2 + (kz-zcen)**2))


# multiply our noisy fourier transform by the gaussian 
filtered_signal = []

for k in range(0, 49):
    filtered = (gaussian * fft_tot_signal[k])
    filtered = np.fft.ifftn(np.fft.ifftshift(filtered))
    filtered_signal.append(filtered)


# plot the final whatever
plot_data = []
for j in range(0,49,3):

  signal = filtered_signal[j]
  normal_sig_abs = np.abs(signal)/np.abs(signal).max()

  # generate data for isosurface of the 3D data 
  fig_data = go.Isosurface( x = xv.flatten(), y = yv.flatten(), z = zv.flatten(),
                           value = normal_sig_abs.flatten(), isomin=0.6, isomax=1)

  plot_data.append(fig_data)

fig=go.Figure(data=plot_data)
fig.update_layout(
    title='Cleaned Signal Data',
    scene=dict(
        xaxis=dict(title = 'X',nticks=5, tickfont=dict(size=15),titlefont=dict(size=30)),
        yaxis=dict(title = 'Y',nticks=5, tickfont=dict(size=15),titlefont=dict(size=30)),
        zaxis=dict(title = 'Z',tickvals= [-5,0,5,10], tickfont=dict(size=15),titlefont=dict(size=30))
    )
)
fig.show()


## PART 3: DETERMINE AND PLOT THE X-Y COORDINATES OF THE SUBPATH 


# from the denoised data: 

# find the corresponding locations
location = []
for t in range(49):
   max = np.unravel_index(np.argmax(filtered_signal[t]), np.shape(filtered_signal[t]))
   location.append(max)

# get the x and y of these locations
xloc = []
yloc = []
for t in range(49):
    xloc.append(location[t][0])
    yloc.append(location[t][1])

# rescale x and y axes
xloc = x[xloc]
yloc = y[yloc]

# plot 2d path!
fig, ax = plt.subplots()

ax.plot(xloc, yloc)

ax.set_xbound(-10, 10)
ax.set_ybound(-10, 10)
#plt.title("Top-Down View of Submarine's Path")
plt.xlabel("X", fontsize = 25)
plt.ylabel("Y", fontsize = 25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()