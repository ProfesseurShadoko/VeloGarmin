

from fancy_package import Message, Task, ProgressBar, cstr
import numpy as np
import pandas as pd




class BikeRide:
    """
    All values must be given in SI units.
    Simple class that contains the code to process a bike ride (compute power, drag, remove pauses).
    
    Columns:
    --------
        self.data['time'] : datetime
        self.data['lat'] : float
        self.data['lon'] : float
        self.data['distance'] : float (in m, distance traveled from start)
        self.data['altitude'] : float (in m)
        self.data['heart_rate'] : float (in bpm)
        self.data['speed'] : float (in m/s)
        self.data['rho'] : float (in kg/m^3, air density)
        self.data['drag'] : float (in N, drag force)
        self.data['kinetic_energy'] : float (in J, kinetic energy)
        self.data['potential_energy'] : float (in J, potential energy)
        self.data['delta_time'] : delta_time (time between two consecutive points)
        self.data['delta_distance'] : float (in m, distance between two consecutive points)
        self.data['delta_altitude'] : float (in m, altitude between two consecutive points)
        self.data['slope'] : float (1 means 100%, slope between two consecutive points)
        self.data['delta_time_seconds'] : float (in s, time between two consecutive points)
        self.data['cumulative_time_seconds'] : float (in s, cumulative time since the start, paused removed)
        self.data['gravity_power'] : float (in W, power due to gravity when going up)
        self.data['kinetic_power'] : float (in W, change in speed, sum of other powers)
        self.data['drag_power'] : float (in W, power due to drag)
        self.data['watts'] : float (in W, total power, deduced from 3 preceding columns, by watt = kinetic - drag - gravity)
    
    Methods:
    --------
    The following method might be useful for data access, rather than accessing the whole dataframe:
    
        get_data(columns:List[str]) -> pd.DataFrame : returns a deep copy of the dataframe with only the columns specified.
    """
    
    def __init__(self, data:pd.DataFrame, mass:float, height:float, bike_mass:float):
        """
        All values must be given in SI units.
        """
        self.data = data.copy(deep=True)
        self.mass = mass
        assert height < 5, "Size must be given in meters, not cm!"
        self.size = height
        self.bike_mass = bike_mass
        self.original_data = data.copy(deep=True)
        
        
        with Task("Processing data", new_line=True):
            
            ##################
            ### CHECK DATA ###
            ##################
            
            with Task("Loading data", new_line=False):
                required_columns = ["time", "lat", "lon", "distance", "altitude", "heart_rate", "speed"]
                for col in required_columns:
                    if col not in self.data.columns:
                        Message(f"Missing column {cstr(col):r}", "!")
                        raise ValueError(f"Missing column {col}")
                    
                self.data["time"] = pd.to_datetime(self.data["time"])
            
            #################
            ### CONSTANTS ###
            #################
            
            drag_coefficient = 1.0
            rho0 = 1.225 # kg/m^3 --> air volumic mass at 0m
            L = 0.0065 # K/m --> temperature gradient
            T0 = 288.15 # K --> temperature at 0m
            R = 287.05 # J/(kg*K) --> gas constant
            g = 9.807 # m/s^2 --> gravity


            #######################
            ### ABSOLUTE VALUES ###
            #######################
            
            with Task("Computing drag", new_line=False):
                projected_frontal_area = 0.0293 * (self.size**0.725) * (mass**0.425) + 0.0604
                self.data['rho'] = rho0 * (1 - L * self.data['altitude'] / T0)**(g / (R * L - 1))
                kinetic_pressure = 0.5 * self.data['rho'] * self.data['speed']**2
                self.data['drag'] = drag_coefficient * projected_frontal_area * kinetic_pressure
                
            with Task("Computing kinetic energy", new_line=False):
                self.data['kinetic_energy'] = 0.5 * (self.mass + self.bike_mass) * self.data['speed']**2
            with Task("Computing potential energy", new_line=False):
                self.data['potential_energy'] = (self.mass + self.bike_mass) * 9.81 * self.data['altitude']
            
            
            ####################
            ### DELTA VALUES ###
            ####################
            
            with Task("Computing delta values", new_line=False):
                for col in ["time", "distance", "altitude", "speed", "kinetic_energy", "potential_energy"]:
                    self.data[f"delta_{col}"] = self.data[col].diff()
                self.data = self.data.iloc[1:].copy(deep=True) # first column
                
                self.data['slope'] = self.data['delta_altitude'] / self.data['delta_distance'] # 100 * slope is slope in %
            self.data["delta_time_seconds"] = self.data["delta_time"].dt.total_seconds()
            
            
            ############################
            ### PAUSE IDENTIFICATION ###
            ############################
            
            with Task("Removing pauses from the ride", new_line=True):
                
                intitial_length = len(self.data)
                mask_to_remove = (
                    (self.data['delta_distance'] < 1) & (self.data["delta_time_seconds"] > 10) # no movement (less than 1m) over 10s.
                ) | (
                    (self.data['delta_distance'] < 1) & (self.data["delta_distance"] < self.data["delta_time_seconds"] * 1) # no movement (less than 10cm) over and average speed of less than 3.6 km/h
                )
                self.data = self.data[~mask_to_remove].copy(deep=True)
                
                # remove data where speed, distance delta and time delta don't match
                intermediate_length = len(self.data)
                mask_to_remove = (
                    (self.data["delta_distance"] - self.data["delta_time_seconds"] * self.data["speed"]).abs() > self.data["delta_distance"] / 2 # tolerate 50% error
                )
                self.data = self.data[~mask_to_remove].copy(deep=True)
            
                final_length = len(self.data)
                Message(f"Removed {cstr((intitial_length - final_length)/intitial_length, format_spec='.2%'):y} or the data", "?")
            self.data["cumulative_time_seconds"] = self.data["delta_time_seconds"].cumsum()
            
            
            #########################
            ### POWER COMPUTATION ###
            #########################
            
            with Task("Computing Watts", new_line=False):
                energy_delta = self.data["delta_kinetic_energy"] + self.data["delta_potential_energy"]
                drag_power = - self.data["drag"] * self.data["speed"]
                
                self.data["watts"] = energy_delta / self.data["delta_time_seconds"] - drag_power # power(pedalage) + power(drag) = energy / time
                self.data["watts"] = self.data["watts"].rolling(window=5, min_periods=1, center=True).median() # smooth the power
                # were wattis nan, set to 0
                self.data.loc[self.data["watts"].isna(), "watts"] = 0
                self.data.loc[self.data["watts"] < 0, "watts"] = 0 # no negative power # remove breaking
                
            # finally we do as if the pauses were never there
            self.data.reset_index(drop=True, inplace=True)
            
            
            ##############################
            ### SPEED AJUSTED TO SLOPE ###
            ##############################
            
            with Task("Computing speed adjusted to slope", new_line=False):
                self.data['adj_speed'] = (
                    self.data['speed']**2 * self.data['watts'] / self.data['drag']
                )**(1/3)
                self.data.loc[
                    self.data['adj_speed'] < self.data['speed'], 'adj_speed'
                ] = np.nan # remove values in decent where speed is higher than it should be
                
            # let's try to make them equal on flat
            data_flat = self.data[self.data['slope'].abs() < 0.01]
            speed_ratio = data_flat['speed'].mean() / data_flat['adj_speed'].mean()
            self.data['adj_speed'] = self.data['adj_speed'] * speed_ratio
            

            

            
    def get_data(self, columns:list = None) -> pd.DataFrame:
        """
        Returns a deep copy of the dataframe.
        """
        if columns is None:
            return self.data.copy(deep=True)
        else:
            return self.data[columns].copy(deep=True)
            
            
            
        