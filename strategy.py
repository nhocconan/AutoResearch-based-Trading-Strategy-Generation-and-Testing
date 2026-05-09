#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on daily close for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 20-period average volume on daily
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(high_max[i]) or
            np.isnan(low_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_1d_aligned[i]
        vol_avg_val = vol_avg_1d_aligned[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        
        if position == 0:
            # Enter long: Price breaks above upper channel + above daily EMA + volume above average
            if close[i] > upper_channel and close[i] > ema_val and volume[i] > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below lower channel + below daily EMA + volume above average
            elif close[i] < lower_channel and close[i] < ema_val and volume[i] > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below lower channel or below daily EMA
            if close[i] < lower_channel or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above upper channel or above daily EMA
            if close[i] > upper_channel or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on daily close for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 20-period average volume on daily
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(high_max[i]) or
            np.isnan(low_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_1d_aligned[i]
        vol_avg_val = vol_avg_1d_aligned[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        
        if position == 0:
            # Enter long: Price breaks above upper channel + above daily EMA + volume above average
            if close[i] > upper_channel and close[i] > ema_val and volume[i] > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below lower channel + below daily EMA + volume above average
            elif close[i] < lower_channel and close[i] < ema_val and volume[i] > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below lower channel or below daily EMA
            if close[i] < lower_channel or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above upper channel or above daily EMA
            if close[i] > upper_channel or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals