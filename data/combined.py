from env.module import *
from data.boundary_merged import *
from data.grid import *

# Combine along rows
combined = np.concatenate([Boundary_train, Residual_train], axis=0)

# Save as CSV
np.savetxt("combined_dataset.csv", combined, delimiter=",")
print("Dataset saved as combined_dataset.csv")
print("Total data points:", combined.shape[0])


fig = plt.figure(figsize=(7, 7))
plt.scatter(Boundary_train[:,0], Boundary_train[:,1], c='g', marker='o', alpha=0.7)
plt.scatter(Residual_train[:,0], Residual_train[:,1], c='r', marker='.', alpha=0.5)
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.title('Positions of collocation points: Grid and boundary data');
#plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)