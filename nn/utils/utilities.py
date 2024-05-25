import numpy as np
import math

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180.0

def add_noise(df, noise_level=0.01):
    noisy_df = df.copy()
    for column in noisy_df.columns:
        if np.issubdtype(noisy_df[column].dtype, np.number):
            noise = np.random.normal(0, noise_level, noisy_df[column].shape)
            noisy_df[column] = noisy_df[column] + noise
    return noisy_df


def check_format(input):
    """
    Check the format of a given input and extract components.

    The expected format is 'x1:x2:x3' or 'x1 x2 x3',
    where 'x1', 'x2', and 'x3' are separated by a colon (':') or a space (' ').

    Parameters:
        input (str): The input to be checked and parsed.

    Returns:
        False: If the input does not match the expected format.
        list: A list containing the three components extracted from the input if the format is correct.
    """
    separators = [':', ' ']
    separator = None
    for sep in separators:
        if sep in input:
            separator = sep
            break

    if separator is None:
        return False

    components = input.split(separator)

    # Check for correct format
    if len(components) != 3:
        return False
    else:
        return components
    
def hms_to_hours(time_string):
    """
    Converts Hours string to float
    :param time_string: Hours String (hh:mm:ss.ss)
    """        
    # Verify separator
    try:
        components = check_format(time_string)

        if components:
            hours = abs(int(components[0]))
            minutes = int(components[1])
            seconds = float(components[2])

            total_hours = hours + minutes / 60 + seconds / 3600

            sign = -1 if "-" in time_string else 1
            return sign*total_hours
        else:
            return None
    except:
        return None

def dms_to_degrees(degrees_string):
    """
    Converts Degrees string to float
    :param degrees_string: Degrees String (dd:mm:ss.ss)
    """
    # Verify separator
    try:
        components = check_format(degrees_string)

        if components:
            degrees_int = abs(int(components[0]))
            minutes = int(components[1])    
            seconds = float(components[2])

            degrees = degrees_int + minutes / 60 + seconds / 3600

            sign = -1 if "-" in degrees_string else 1
            return sign*degrees
        else:
            return None
    except:
        return None

def is_numeric(input):
    """
    Check if the input is a numeric value.
    Parameters:
        input (int or float): The value to be checked.
    Returns:
        bool: True if the input is numeric (int or float), False otherwise.
    """
    if isinstance(input, (int, float)):
        return True
    else:
        return False
         
def check_exists(path):
    """
    Check if a file or directory exists at the given path.
    Parameters:
        path (str): The path to check for existence.
    Returns:
        bool: True if a file or directory exists at the given path, False otherwise.
    """
    import os
    if os.path.exists(path):
        return True
    else:
        return False

def hours_to_hms(hours, decimal_digits=0):
    """
    Converts Float Hour to string Hour, in format hh:mm:ss:cc
    :param hours: Hours (float)
    """
    if is_numeric(hours):        
        sign = "-" if hours < 0 else ""
        hours = abs(hours)
        whole_hours = int(hours)
        fractional_hours = hours - whole_hours

        minutes = int(fractional_hours * 60)
        fractional_minutes = fractional_hours * 60 - minutes

        seconds = int(fractional_minutes * 60)
        fractional_seconds = fractional_minutes * 60 - seconds

        seconds_str = f"{seconds:02}.{int(fractional_seconds * (10 ** decimal_digits)):02d}"

        time_string = f"{sign}{whole_hours:02}:{minutes:02}:{seconds_str}"
        
        return time_string
    else:
        return None

def degrees_to_dms(degrees):
    """
    Converts Degrees to string, in format dd:mm:ss:cc
    :param hours: Degrees (float)
    """
    if is_numeric(degrees):
        sign = "-" if degrees < 0 else "+"
        degrees = abs(degrees)
        degrees_int = int(degrees)
        minutes = int((degrees - degrees_int) * 60)
        seconds = int(((degrees - degrees_int) * 60 - minutes) * 60)
        seconds_decimal = int((((degrees - degrees_int) * 60 - minutes) * 60 - seconds) * 100)

        # Formated value
        degrees_string = f'{sign}{degrees_int:02}:{minutes:02}:{seconds:02}.{seconds_decimal:02}'

        return degrees_string
    else:
        return None
    
def get_elevation_azimuth(ha, dec, latitude):
    """Calculates Azimuth and Elevation
    params: HourAngle, Declination, Latitude
    return: Elevation, Azimuth
    """ 
    if not is_numeric(ha):
        ha = hms_to_hours(ha)
    if not is_numeric(dec):
        dec = dms_to_degrees(dec)
    if not is_numeric(latitude):
        latitude = dms_to_degrees(latitude)

    H = ha * 15

    #altitude calc
    sinAltitude = (math.sin(dec * DEG2RAD)) * (math.sin(latitude * DEG2RAD)) + (math.cos(dec * DEG2RAD) * math.cos(latitude * DEG2RAD) * math.cos(H * DEG2RAD))
    elevation = math.asin(sinAltitude) * RAD2DEG #altura em graus
    elevation = round(elevation, 2)

    #azimuth calc
    y = -1 * math.sin(H * DEG2RAD)
    x = (math.tan(dec * DEG2RAD) * math.cos(latitude * DEG2RAD)) - (math.cos(H * DEG2RAD) * math.sin(latitude * DEG2RAD))

    #This AZposCalc is the initial AZ for dome positioning
    azimuth = math.atan2(y, x) * RAD2DEG

    #converting neg values to pos
    if (azimuth < 0) :
        azimuth = azimuth + 360    

    return elevation, azimuth