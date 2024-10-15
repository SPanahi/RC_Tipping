import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rc



class aux_funcs:
    def __init__():
        print('Start')
    
    
def amoc_2d(A = 1.33, m = 0, lambda0 = -2.7, t0 = 0, tf = 150, dt = 8e-2, Hmax = 0.15, coeff1 = 3e-1):
        # Define parameters
        Hmin = lambda0
        r = (Hmax - Hmin) / (tf - t0)

        # Initial conditions
        x0 = np.array([1.7, lambda0])

        # Noise strength
        sigma = np.array([coeff1, 0])

        # Heun method implementation
        def ode2_heun(f, g, t, x0, sigma):
            d = len(x0)  # System dimension
            x = np.zeros((d, len(t)))  # Output time series
            x[:, 0] = x0  # Set initial condition
            h = t[1] - t[0]  # Temporal step length
            
            for step_i in range(len(t) - 1):
                w = np.sqrt(h) * sigma * np.random.randn(d)
                y = x[:, step_i] + h * f(t[step_i], x[:, step_i]) + g(x[:, step_i]) * w
                x_new = x[:, step_i] + (h / 2) * (f(t[step_i], x[:, step_i]) + f(t[step_i], y)) + (g(x[:, step_i]) + g(y)) / 2 * w
                x[:, step_i + 1] = x_new

            return t, x

        # Define the AMOC2d system
        def amoc2d(t, X, A, m, Hmax, lambda0, r):
            x = X[0]
            param = X[1]
            
            dx1 = -A * (x - m) ** 2 - param
            if Hmax >= param >= lambda0:
                dx2 = r
            else:
                dx2 = 0
            
            return np.array([dx1, dx2])

        # Define the noise function g (identity in this case)
        def eq_pp_mdn(x):
            # g = np.sqrt(x)  # Uncomment this if you want colored noise
            g = 1  # As per the modified function in MATLAB
            return np.ones_like(x) * g


        # Time vector
        t = np.arange(t0, tf, dt)

        # Solve the ODE using the Heun method
        t, X = ode2_heun(lambda t, x: amoc2d(t, x, A, m, Hmax, lambda0, r), eq_pp_mdn, t, x0, sigma)

        x = X[0, :]  # data
        x[x < -4] = -4  # Clipping x values below -4
        param = X[1, :]  # parameter
        # Find when param crosses zero
        l1 = param[(param > 0) & (param < 5e-3)]
        rl, clf = np.where(param == l1[0])[0][0], np.where(param == l1[0])[0]

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # First subplot: x vs time
        ax1.plot(t + 1924, x, linewidth=5)
        ax1.set_xlabel('Time', fontsize=16)
        ax1.set_ylabel('x', fontsize=16)
        ax1.set_xlim([1924, 2074])
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.axvline(x=t[clf] + 1924, linewidth=3, linestyle='--', color='k')


        # Second subplot: lambda vs time
        ax2.plot(t + 1924, param, linewidth=5)
        ax2.axhline(y=0, linewidth=3, linestyle='--', color='k')
        ax2.set_xlabel('Time', fontsize=16)
        ax2.set_ylabel(r'$\lambda$', fontsize=16)
        ax2.set_xlim([1924, 2074])
        ax2.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        plt.show()
        return np.transpose(X)
    
    
    
def d2pkl(name, data):
    dim = np.shape(data)[1]
    df = pd.DataFrame(data)
    df.to_pickle(f'data/{name}.pkl')
    
    
    
def data_gen(name):
    class_object = globals()[name]
    obj = class_object()
    d2pkl(name, obj)

