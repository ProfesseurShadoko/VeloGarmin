

from .bikeride import BikeRide



def min_periods(bk:BikeRide, minutes:int=0, seconds:int=0):
    """
    Returns:
        int: 95% confidence upper bound of time delta. Used to define min_periods with rolling
    """
    data = bk.get_data(columns=['delta_time_seconds'])
    tau = data["delta_time_seconds"].mean() + data["delta_time_seconds"].std()*2
    return int((60*minutes + seconds) / tau)


def get_ftp(bk:BikeRide) -> float:
    """
    FTP = 95% of the maximal 20 minutes power.
    """
    data = bk.get_data(columns=['time', 'watts']).set_index('time')
    return data['watts'].rolling(window='20min', min_periods=min_periods(bk, minutes=20)).mean().max() * 0.95

def get_ftp_per_kg(bk:BikeRide) -> float:
    """
    FTP/kg = 95% of the maximal 20 minutes power divided by the rider weight.
    """
    return get_ftp(bk) / bk.mass

def get_ppo(bk:BikeRide) -> float:
    """
    PPO = 95% of the maximal 5 minutes power.
    """
    data = bk.get_data(columns=['time', 'watts']).set_index('time')
    return data['watts'].rolling(window='150s', min_periods=min_periods(bk, seconds=150)).mean().max()

def get_v02max(bk:BikeRide) -> float:
    """
    Returns:
        VO_{2 max}: (mL/kg/min)
    """
    return (0.01141*get_ppo(bk) + 0.435) * 1_000 / bk.mass


def get_normalized_power(bk:BikeRide) -> float:
    data = bk.get_data(columns=['watts'])
    return (data["watts"]**4).mean()**(1/4)

def get_intensity_factor(bk:BikeRide) -> float:
    return get_normalized_power(bk) / get_ftp(bk)

def get_training_stress_score(bk:BikeRide, absolute:bool = False) -> float:
    data = bk.get_data(columns=['cumulative_time_seconds'])
    if not absolute:
        return data["cumulative_time_seconds"].max() * get_normalized_power(bk) * get_intensity_factor(bk) / (get_ftp(bk) * 3600) * 100
    else:
        return data["cumulative_time_seconds"].max() * get_normalized_power(bk) * get_intensity_factor(bk) * get_ftp(bk) / 36 / (3.5*bk.mass)**2

