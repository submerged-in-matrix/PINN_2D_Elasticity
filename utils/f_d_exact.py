from env.module import *
from utils.material_specs import *

def u_x_exact(x, y):
    u_x_assigned = tf.cos(2*pi*x) * tf.sin(pi*(y))                                         
    # utemp = tf.cos(2*pi*(x-xmin)/(xmax-xmin)) * tf.sin(pi*(y-ymin)/(ymax-ymin))    
    return u_x_assigned
def u_y_exact(x, y):
    u_y_assigned = tf.sin(pi*x) * Q / 4 * tf.pow(y,4)                                         
    # utemp = tf.sin(pi*(x-xmin)/(xmax-xmin)) * Q/4*tf.pow((y-ymin)/(ymax-ymin),4)
    return u_y_assigned
def f_x_exact(x,y):
    f_x = 1.0*(-4*tf.pow(pi,2)*tf.cos(2*pi*x)*tf.sin(pi*y)+pi*tf.cos(pi*x)*Q*tf.pow(y,3))+\
    0.5*(-9*tf.pow(pi,2)*tf.cos(2*pi*x)*tf.sin(pi*y)+pi*tf.cos(pi*x)*Q*tf.pow(y,3))           
    # gtemp = 1.0*(-4*tf.pow(pi,2)*tf.cos(2*pi*(x-xmin)/(xmax-xmin))*tf.sin(pi*(y-ymin)/(ymax-ymin))+pi*tf.cos(pi*(x-xmin)/(xmax-xmin))*Q*tf.pow((y-ymin)/(ymax-ymin),3))+\
    # 0.5*(-9*tf.pow(pi,2)*tf.cos(2*pi*(x-xmin)/(xmax-xmin))*tf.sin(pi*(y-ymin)/(ymax-ymin))+pi*tf.cos(pi*(x-xmin)/(xmax-xmin))*Q*tf.pow((y-ymin)/(ymax-ymin),3))
    return f_x
def f_y_exact(x,y):
    f_y = lmda*(3*tf.sin(pi*x)*Q*tf.pow(y,2)-2*tf.pow(pi,2)*tf.sin(2*pi*x)*tf.cos(pi*y))+\
            mu*(6*tf.sin(pi*x)*Q*tf.pow(y,2)-2*tf.pow(pi,2)*tf.sin(2*pi*x)*tf.cos(pi*y)-tf.pow(pi,2)*tf.sin(pi*x)*Q*tf.pow(y,4)/4)
    # gtemp = lmda*(3*tf.sin(pi*(x-xmin)/(xmax-xmin))*Q*tf.pow((y-ymin)/(ymax-ymin),2)-2*tf.pow(pi,2)*tf.sin(2*pi*(x-xmin)/(xmax-xmin))*tf.cos(pi*(y-ymin)/(ymax-ymin)))+\
    #     mu*(6*tf.sin(pi*(x-xmin)/(xmax-xmin))*Q*tf.pow((y-ymin)/(ymax-ymin),2)-2*tf.pow(pi,2)*tf.sin(2*pi*(x-xmin)/(xmax-xmin))*tf.cos(pi*(y-ymin)/(ymax-ymin))-tf.pow(pi,2)*tf.sin(pi*(x-xmin)/(xmax-xmin))*Q*tf.pow((y-ymin)/(ymax-ymin),4)/4)
    return f_y