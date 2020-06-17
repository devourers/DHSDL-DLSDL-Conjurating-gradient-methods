import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize





def J(x):
    a = [-0.25, 0.75, 0.5, -0.25, 0.25]
    b = [-2, 4, -1, 2, -2]
    x = np.array(x)
    a = np.array(a)
    b = np.array(b)
    res = (np.dot(a, x))**4
    res += np.linalg.norm(x-b)**2
    return res

a = [-0.25, 0.75, 0.5, -0.25, 0.25]
b = [-2, 4, -1, 2, -2]
a = np.array(a)
b = np.array(b)
print("[a, b]", np.dot(a,b))
print("norm of a", np.linalg.norm(a))
print("norm of b", np.linalg.norm(b))


def dJdx1(x):
    delta = [0.0001, 0, 0, 0, 0]
    delta = np.array(delta)
    return (J(x+delta) - J(x))/0.0001

def dJdx2(x):
    delta = [0, 0.00001, 0, 0, 0]
    delta = np.array(delta)
    return (J(x+delta) - J(x))/0.00001

def dJdx3(x):
    delta = [0, 0, 0.00001, 0, 0]
    delta = np.array(delta)
    return (J(x+delta) - J(x))/0.00001

def dJdx4(x):
    delta = [0, 0, 0, 0.00001, 0]
    delta = np.array(delta)
    return (J(x+delta) - J(x))/0.00001

def dJdx5(x):
    delta = [0, 0, 0, 0, 0.00001]
    delta = np.array(delta)
    return (J(x+delta) - J(x))/0.00001


def GradJ(x):
    return np.array([dJdx1(x), dJdx2(x), dJdx3(x), dJdx4(x), dJdx5(x)])




# g_k = g, g_{k-1} = g0 
# y_k = y, y_{k-1} = y0  
# a_k = alpha, a_{k-1} = alpha0
def CJM_DHSDL(k):
    results =[]
    mu = 1.1
    t = 0.1
    x0 = [1, 1, 1, 1, 1]
    g0 = GradJ(x0)
    d0 = -1 * g0
    #alpha0 = checkAlpha(x0, 10, d0, g0)
    alpha0 = 0.2
    x = x0 + alpha0*d0
    results.append(J(x))
    g = GradJ(x)
    s0 = x - x0
    y0 = g - g0
    b = (np.linalg.norm(g)**2 - (np.linalg.norm(g)/np.linalg.norm(g0))*abs(np.dot(np.transpose(g), g0)))/((mu*abs(np.dot(np.transpose(g), d0))) + np.dot(np.transpose(d0), y0)) - t*(np.dot(np.transpose(g), s0))/(np.dot(np.transpose(d0), y0))
    #alpha0 = scipy.optimize.line_search(J, GradJ, x0, y0)[0]
    #print(scipy.optimize.line_search(J, GradJ, x0, y0)[0])
    #have b_1, x_1, x_0, g_1, g_0, d_0, s_0, y_0
    for i in range (k-1):
        temp_x = x
        temp_g = g
        d = -1 * g + b*d0
        #alpha0 = checkAlpha(x, 1, d, g)
        x += alpha0*d
        results.append(J(x))
        #have x2, d1
        g0 = temp_g
        x0 = temp_x
        g = GradJ(x)
        d0 = d
        #have g2, x1, g1
        s0 = x - x0
        y0 = g - g0    
        b = (np.linalg.norm(g)**2 - (np.linalg.norm(g)/np.linalg.norm(g0))*abs(np.dot(np.transpose(g), g0)))/((mu*abs(np.dot(np.transpose(g), d0))) + np.dot(np.transpose(d0), y0)) - t*(np.dot(np.transpose(g), s0))/(np.dot(np.transpose(d0), y0))  
        #alpha0 = scipy.optimize.line_search(J, GradJ, x0, y0)[0]
        #print(scipy.optimize.line_search(J, GradJ, x0, y0)[0])
    return results



# g_k = g, g_{k-1} = g0 
# y_k = y, y_{k-1} = y0  
# a_k = alpha, a_{k-1} = alpha0   
def CJM_DLSDL(k):
    results =[]
    mu = 1.1
    t = 0.1
    x0 = [-1, -1, -1, -1, -1]
    g0 = GradJ(x0)
    d0 = -1 * g0
    #alpha0 = checkAlpha(x0, 10, d0, g0)
    alpha0 = 0.2
    x = x0 + alpha0*d0
    results.append(J(x))
    g = GradJ(x)
    s0 = x - x0
    y0 = g - g0
    b = (np.linalg.norm(g)**2 - (np.linalg.norm(g)/np.linalg.norm(g0))*abs(np.dot(np.transpose(g), g0)))/((mu*abs(np.dot(np.transpose(g), d0))) - np.dot(np.transpose(d0), g0)) - t*(np.dot(np.transpose(g), s0))/(np.dot(np.transpose(d0), y0))
    #have b_1, x_1, x_0, g_1, g_0, d_0, s_0, y_0
    #alpha0 = scipy.optimize.line_search(J, GradJ, x0, y0)[0]
    for i in range (k-1):
        temp_x = x
        temp_g = g
        d = -1 * g + b*d0
        #alpha0 = checkAlpha(x, alpha0, d, g)
        x += alpha0*d
        results.append(J(x))
        #have x2, d1
        g0 = temp_g
        x0 = temp_x
        g = GradJ(x)
        d0 = d
        #have g2, x1, g1
        s0 = x - x0
        y0 = g - g0    
        b = (np.linalg.norm(g)**2 - (np.linalg.norm(g)/np.linalg.norm(g0))*abs(np.dot(np.transpose(g), g0)))/((mu*abs(np.dot(np.transpose(g), d0))) - np.dot(np.transpose(d0), g0)) - t*(np.dot(np.transpose(g), s0))/(np.dot(np.transpose(d0), y0))
        #alpha0 = scipy.optimize.line_search(J, GradJ, x0, y0)[0]
        #print(scipy.optimize.line_search(J, GradJ, x0, y0)[0])
    return results


 
results_DHSDL = CJM_DHSDL(30)
results_DLSDL = CJM_DLSDL(30)
results_DHSDL = np.array(results_DHSDL)
results_DLSDL = np.array(results_DLSDL)

x_plot = np.arange(1, 31, 1)

x_target = []
res_check = scipy.optimize.minimize(J, [0, 0, 0, 0, 0], method = 'CG')


for i in range(30):
    x_target.append(res_check.fun)
    
x_target = np.array(x_target)

print(res_check)
print("Result of DHSDL", results_DHSDL[-1], "in 30 steps, delta with real solution is", x_target[0] - res_check.fun)
print("Result of DLSDL", results_DLSDL[-1], "in 30 steps, delta with real solution is", x_target[0] - res_check.fun)

print("min point is via analytical", -1.1165 * a + b)
print("analytical solution", J(-1.1165  * a + b))
    

plt.plot(x_plot, results_DHSDL)
plt.plot(x_plot, results_DLSDL)
plt.plot(x_plot, x_target)
plt.show()
        
    