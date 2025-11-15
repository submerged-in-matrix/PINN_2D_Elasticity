from env.module import *
from data.grid import *
from data.boundary_merged import *

Residual_train_orig = np.array(Residual_train)
Boundary_list_orig = [np.array(x) for x in Boundary_list]

Dirichlet_up_orig = np.array(Dirichlet_up_train)
Dirichlet_x_low_orig = np.array(Dirichlet_x_low_train)
Dirichlet_y_low_orig = np.array(Dirichlet_y_low_train)
Dirichlet_right_orig = np.array(Dirichlet_right_train)
Dirichlet_left_orig = np.array(Dirichlet_left_train)
Neumann_right_orig = np.array(Neumann_right_train)
Neumann_left_orig = np.array(Neumann_left_train)
Neumann_up_orig = np.array(Neumann_up_train)

# --------------- SPLIT (always from *_full arrays) -----------------------
Residual_train, Residual_val = train_test_split(Residual_train_orig, test_size=0.2, random_state=42)

# Split boundary coordinates
x_up_orig, x_lo_orig, x_ri_orig, x_le_orig = Boundary_list_orig
x_up_train, x_up_val = train_test_split(x_up_orig, test_size=0.2, random_state=42)
x_lo_train, x_lo_val = train_test_split(x_lo_orig, test_size=0.2, random_state=42)
x_ri_train, x_ri_val = train_test_split(x_ri_orig, test_size=0.2, random_state=42)
x_le_train, x_le_val = train_test_split(x_le_orig, test_size=0.2, random_state=42)

Boundary_list_train = [x_up_train, x_lo_train, x_ri_train, x_le_train]
Boundary_list_val   = [x_up_val,  x_lo_val,  x_ri_val,  x_le_val]

# Dirichlet_x: up + x_low
Dirichlet_up_train, Dirichlet_up_val = train_test_split(Dirichlet_up_orig, test_size=0.2, random_state=42)
Dirichlet_x_low_train, Dirichlet_x_low_val = train_test_split(Dirichlet_x_low_orig, test_size=0.2, random_state=42)
Dirichlet_x_train = np.concatenate([Dirichlet_up_train, Dirichlet_x_low_train], axis=0)
Dirichlet_x_val   = np.concatenate([Dirichlet_up_val, Dirichlet_x_low_val], axis=0)

# Dirichlet_y: y_low + right + left
Dirichlet_y_low_train, Dirichlet_y_low_val = train_test_split(Dirichlet_y_low_orig, test_size=0.2, random_state=42)
Dirichlet_right_train, Dirichlet_right_val = train_test_split(Dirichlet_right_orig, test_size=0.2, random_state=42)
Dirichlet_left_train, Dirichlet_left_val = train_test_split(Dirichlet_left_orig, test_size=0.2, random_state=42)
Dirichlet_y_train = np.concatenate([Dirichlet_y_low_train, Dirichlet_right_train, Dirichlet_left_train], axis=0)
Dirichlet_y_val   = np.concatenate([Dirichlet_y_low_val,   Dirichlet_right_val,   Dirichlet_left_val], axis=0)

# Neumann_xx: right + left
Neumann_right_train, Neumann_right_val = train_test_split(Neumann_right_orig, test_size=0.2, random_state=42)
Neumann_left_train, Neumann_left_val = train_test_split(Neumann_left_orig, test_size=0.2, random_state=42)
Neumann_xx_train = np.concatenate([Neumann_right_train, Neumann_left_train], axis=0)
Neumann_xx_val   = np.concatenate([Neumann_right_val,   Neumann_left_val], axis=0)

# Neumann_yy: up
Neumann_up_train, Neumann_up_val = train_test_split(Neumann_up_orig, test_size=0.2, random_state=42)
Neumann_yy_train = Neumann_up_train
Neumann_yy_val   = Neumann_up_val

# -------------- Print for sanity check --------------
print("Residual_train shape:", Residual_train.shape)
print("Dirichlet_x_train shape:", Dirichlet_x_train.shape)
print("Dirichlet_y_train shape:", Dirichlet_y_train.shape)
print("Neumann_xx_train shape:", Neumann_xx_train.shape)
print("Neumann_yy_train shape:", Neumann_yy_train.shape)