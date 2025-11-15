from env.module import *
from utils.precision import DTYPE
# constants & model parameters
pi = tf.constant(np.pi, dtype=DTYPE)
E = tf.constant(4e11/3, dtype=DTYPE)                # Young's modulus. roughly 133 GPa, a realistic value for engineering materials.
v = tf.constant(1/3, dtype=DTYPE)                   # Poisson's ratio. ν = 1/3 is typical for metals (e,g., steel is around 0.3).
E = E/1e11                                          # Normalize Young's modulus to 1e11 for better numerical stability
                                                    # E is set to 4e11/3, which is a common value for materials like concrete or steel, and then normalized by dividing by 1e11.

# Lamé parameters 
lmda = tf.constant(E*v/(1-2*v)/(1+v), dtype=DTYPE)  # linear elasticity theory, specifically Hooke’s Law for isotropic materials in 2D/3D:λ helps model volumetric expansion or compression under stress.
mu = tf.constant(E/(2*(1+v)), dtype=DTYPE)          # shear modulus,describes  deformations under shear stress.y, the stress-strain relations use λ and μ to connect displacements (u) to stresses (σ).

# viscosity = .01/pi
Q = tf.constant(4.0, dtype=DTYPE)                   # a known distributed force (body force), controls the strength of the force applied to the system.

# Dirichlet boundary Conditions. Floating-point numbers are used to define the boundaries of the domain. xmin, xmax, ymin, ymax define the rectangular region where the PDE is solved.
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0