

from fancy_package import Message, Task, ProgressBar, cstr
from .bikeride import BikeRide
from .metrics import min_periods, get_ftp, get_ftp_per_kg, get_ppo, get_v02max
import datetime
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
plt.style.use('dark_background')

#############
### UTILS ###
#############

def smoothen_on_distance(bk:BikeRide, columns:list[str], distance_window:float=2000) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    assert isinstance(columns, list), "columns must be a list of strings"
    
    data = bk.get_data(columns=['distance', *columns])
    dx = data['distance'].diff().median()
    window = int(distance_window / dx)
    
    for column in columns:
        data[column] = data[column].interpolate(method='linear')
        data[column] = savgol_filter(data[column], window, 3)
    return data

##################
### STATISTICS ###
##################

def print_ride_stats(bk: BikeRide):
    Message("Statistiques de la sortie vélo:")
    with Message.tab():
        
        # total duration
        Message.print(
            "Durée totale de la sortie: " + cstr(
                datetime.timedelta(seconds=bk.data['cumulative_time_seconds'].iloc[-1])
            ).bold().cyan()
        )
        
        # total distance
        Message.print(
            "Distance totale parcourue: " + cstr(
                round(bk.data['distance'].iloc[-1] / 1000, 1)
            ).bold().cyan() + " km"
        )
        
        # total elevation gain # for a smoothed one over 200m resolution
        window_length = 200
        dx = bk.data['distance'].diff().median()
        window = int(window_length / dx)
        
        smooth_altitude_gain = np.diff(savgol_filter(bk.data['altitude'], window, 3))
        Message.print(
            "Dénivelé positif: " + cstr(
                round(smooth_altitude_gain[smooth_altitude_gain > 0].sum())
            ).bold().cyan() + " m"
        )
        
        print()
        
        # average speed
        Message.print(
            "Vitesse moyenne: " + cstr(
                round(bk.data['speed'].mean() * 3.6, 1)
            ).bold().cyan() + " km/h"
        )
        
        # Max speed
        Message.print(
            "Vitesse maximale: " + cstr(
                round(bk.data['speed'].max() * 3.6, 1)
            ).bold().cyan() + " km/h"
        )
        
        # Slope Adjusted Average Speed
        Message.print(
            "Vitesse moyenne ajustée à la pente: " + cstr(
                round(bk.data['adj_speed'].mean() * 3.6, 1)
            ).bold().cyan() + " km/h"
        )
            
        
        # Average Corrected Speed
        #Message.print(
        #    "Vitesse moyenne ajustée à la pente: " + cstr(
        #        round(bk.data['corrected_speed'].mean() * 3.6, 1)
        #    ).bold().cyan() + " km/h"
        #)
        
        print()
        
        # Average power
        Message.print(
            "Puissance moyenne: " + cstr(
                round(bk.data['watts'].mean())
            ).bold().cyan() + " W"
        )
        
        # FTP
        Message.print(
            "Functionnal Treshold Power (FTP): " + cstr(
                round(get_ftp_per_kg(bk), 2)
            ).bold().cyan() + " W/kg"
        )
        
        # PPO
        Message.print(
            "Peak Power Output (PPO): " + cstr(
                round(get_ppo(bk))
            ).bold().cyan() + " W"
        )
        
        print()
        
        # Average Heart Rate
        Message.print(
            "Fréquence cardiaque moyenne: " + cstr(
                round(bk.data['heart_rate'].mean())
            ).bold().cyan() + " bpm"
        )
        
        # VO2 max
        Message.print(
            "VO2 max: " + cstr(
                round(get_v02max(bk))
            ).bold().cyan() + " mL/kg/min"
        )
        
        # TSS
        Message.print(
            "Training Stress Score (TSS): " + cstr(
                round(get_v02max(bk))
            ).bold().cyan()
        )
        


######################
### ADJUSTED SPEED ###
######################

def plot_adj_speed(bk:BikeRide):
    plt.figure(figsize=(15, 5))
    plt.title("Comparaison de la vitesse\net de la vitesse ajustée à la pente")
    data_adj = smoothen_on_distance(bk, ['adj_speed', 'speed'], 2000)
    data = bk.get_data(columns=['distance', 'speed'])
    
    plt.scatter(data['distance'] / 1e3, data['speed']*3.6, label='Vitesse', color='gray', s=3, alpha=0.5, marker='x')

    plt.plot(data_adj['distance'] / 1e3, data_adj['speed']*3.6, label='Vitesse lissée', color='darkblue')
    plt.plot(data_adj['distance'] / 1e3, data_adj['adj_speed']*3.6, label='Vitesse ajustée à la pente', color='pink')
    
    plt.ylabel("Vitesse (km/h)")
    plt.xlabel("Distance (km)")
    plt.legend()
    plt.show()


##################
### POWER - HR ###
##################

def plot_power_hr(bk:BikeRide):
    plt.figure(figsize=(15, 5), dpi=400)
    plt.suptitle("Puissance et fréquence cardiaque")
    data_smooth = smoothen_on_distance(bk, ['watts', 'heart_rate'], 2000)
    data = bk.get_data(columns=['distance', 'watts', 'heart_rate'])
    
    grid = GridSpec(2, 4)
    ax = plt.subplot(grid[0, :2])
    ax.scatter(data['distance'][data['watts']>0] / 1e3, data['watts'][data['watts']>0], color='yellow', s=3, alpha=0.1, marker='x', zorder = 1)
    ax.plot(data_smooth['distance'] / 1e3, data_smooth['watts'], label='Puissance', color='yellow', zorder = 2)
    ax.set_ylabel("Puissance (W)")
    
    ax = plt.subplot(grid[1, :2])
    ax.scatter(data['distance'] / 1e3, data['heart_rate'], color='red', s=3, alpha=0.2, marker='x', zorder=1)
    ax.plot(data_smooth['distance'] / 1e3, data_smooth['heart_rate'], label='Fréquence cardiaque', color='red', zorder=2)
    ax.set_ylabel("Fréquence cardiaque (bpm)")
    ax.set_xlabel("Distance (km)")
    
    
    data = data[
        data['watts'] > 0
    ]
    
    ax = plt.subplot(grid[:, 2:])
    sm = ax.scatter(data['watts'], data['heart_rate'], c=data['distance'] / 1e3, s=3, alpha=0.5, cmap='viridis')
    
    # set y and x lim to 95 quantile
    ax.set_ylim(ax.get_ylim()[0], data['heart_rate'].quantile(0.95))
    ax.set_xlim(0, data['watts'].quantile(0.95))
    
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Distance (km)')
    
    ax.set_xlabel("Puissance (W)")
    ax.set_ylabel("Fréquence cardiaque (bpm)")
    plt.tight_layout()
    plt.show()
    
    
    
#####################
### SLOPE - POWER ###
#####################

def plot_slope_power(bk:BikeRide):
    plt.figure(figsize=(15, 5), dpi=400)
    plt.suptitle("Puissance et pente")
    data_smooth = smoothen_on_distance(bk, ['watts', 'slope'], 2000)
    data = bk.get_data(columns=['distance', 'watts', 'slope'])
    
    data_smooth['slope'] *= 100
    data['slope'] *= 100
    
    grid = GridSpec(2, 4)
    ax = plt.subplot(grid[0, :2])
    ax.scatter(data['distance'][data['watts']>0] / 1e3, data['watts'][data['watts']>0], color='yellow', s=3, alpha=0.1, marker='x', zorder = 1)
    ax.plot(data_smooth['distance'] / 1e3, data_smooth['watts'], label='Puissance', color='yellow', zorder = 2)
    ax.set_ylabel("Puissance (W)")
    
    ax = plt.subplot(grid[1, :2])
    #ax.scatter(data['distance'] / 1e3, data['slope'], color='red', s=3, alpha=0.2, marker='x', zorder=1)
    data_smooth[data_smooth['slope']<0] = np.nan
    ax.plot(data_smooth['distance'] / 1e3, data_smooth['slope'], label='Pente', color='purple', zorder=2)
    ax.set_ylabel("Pente (%)")
    ax.set_xlabel("Distance (km)")
    
    
    data = data[
        (data['watts'] > 0) & (data['slope'] > 0)
    ]
    
    ax = plt.subplot(grid[:, 2:])
    sm = ax.scatter(data['watts'], data['slope'], c=data['distance'] / 1e3, s=3, alpha=0.5, cmap='viridis')
    
    # set y and x lim to 95 quantile
    ax.set_ylim(0, data['slope'].quantile(0.95))
    ax.set_xlim(0, data['watts'].quantile(0.95))
    
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Distance (km)')
    
    ax.set_xlabel("Puissance (W)")
    ax.set_ylabel("Pente (%)")
    plt.tight_layout()
    plt.show()



##########################
### POWER DISTRIBUTION ###
##########################

def plot_power_distribution(bk: BikeRide):
    
    grid = GridSpec(5, 1)
    # first lets compute power distribution
    data = bk.get_data(['distance', 'watts'])

    with Task("Computing Gaussian Kernel Density Estimation"):
        kde = gaussian_kde(data['watts'][data['watts']>0].values)
        x_grid = np.linspace(
            data['watts'][data['watts']>0].quantile(0.01),
            data['watts'][data['watts']>0].quantile(0.99),
            400
        )
        kde_values = kde(x_grid)
    
    
    with Task("Performing Gaussian Fit"):
    
        # try fitting gaussians
        def lognormal(x, A, mean, std):
            # return normalized gaussian function
            return A * (1 / (std * x)) * np.exp(-0.5 * ((np.log(x) - mean) / std) ** 2)
        
        
        initial_guess = [1, 100, 100]
        params, _ = curve_fit(lognormal, x_grid, kde_values, p0 = initial_guess)
        A1, m1, s1 = params
    
    # let's plot the result
    plt.figure(figsize=(15,10), dpi=400)
    plt.subplot(grid[:2, :])
    plt.plot(x_grid, kde_values, color='yellow')
    plt.fill_between(x_grid, kde_values, color='yellow', alpha=0.5)
    plt.title("Power distribution over the ride")
    plt.xlabel("Power (W)")
    plt.ylabel("Time")
    plt.yticks([])
    
    
    
    plt.xlim(x_grid.min(), x_grid.max())
    
    # plot fitted gaussians too
    g1 = lognormal(x_grid, A1, m1, s1)
    plt.plot(x_grid, g1, linestyle='--', color='red', label='Lognormal fit', linewidth=2)
    plt.ylim(bottom=0)
    plt.legend()
    
    plt.subplot(grid[2:, :]) # let's plot the fitted lognormal for distributions of first 20%, 20%-40%, etc, 80%-100% of the ride
    
    plt.plot(x_grid, g1, color='red', linestyle='--', zorder = 20)
    cmap = LinearSegmentedColormap.from_list("yellow_to_gray", ["darkblue", "yellow"])
    total_distance = data['distance'].iloc[-1]
    for i in range(5):
        data_local = data[
            data['distance'].between(i * total_distance / 5, (i+1) * total_distance / 5)
        ]
        kde = gaussian_kde(data_local['watts'][data_local['watts']>0].values)
        kde_values = kde(x_grid) / 2 # some normalization so it looks better
        initial_guess = [1, 100, 100]
        params, _ = curve_fit(lognormal, x_grid, kde_values, p0 = initial_guess)
        A1, m1, s1 = params
        g1 = lognormal(x_grid, A1, m1, s1)
        plt.plot(x_grid, g1, color=cmap(i/5), label=f"Distance traveled: {i*20}% - {(i+1)*20}%")
    plt.xlabel("Power (W)")
    plt.xscale('log')
    plt.xticks(
        plt.xticks()[0], plt.xticks()[0].astype(int)
    )
    plt.xlim(x_grid.min(), x_grid.max())
    plt.yticks([])
    plt.ylabel("Time")
    plt.legend()
    plt.tight_layout()
    
    
    plt.show()
    