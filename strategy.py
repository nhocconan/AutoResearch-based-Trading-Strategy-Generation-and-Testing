#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Long when price breaks above 20-period high and 1d EMA34 is rising
# Short when price breaks below 20-period low and 1d EMA34 is falling
# Volume confirmation: current volume > 1.5x 20-period average
# Exit when price crosses the 10-period moving average in the opposite direction
# Targets 50-150 total trades over 4 years (12-37/year) for optimal fee drag

name = "4h_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = pd.Series(ema34_1d).diff().values
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ma_10[i]) or 
            np.isnan(ema34_1d_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ma_10_val = ma_10[i]
        ema34_1d_slope_val = ema34_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above 20-period high, 1d uptrend, volume confirmation
            if close_val > high_20_val and ema34_1d_slope_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: break below 20-period low, 1d downtrend, volume confirmation
            elif close_val < low_20_val and ema34_1d_slope_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 10-period MA or 1d trend turns down
            if close_val < ma_10_val or ema34_1d_slope_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 10-period MA or 1d trend turns up
            if close_val > ma_10_val or ema34_1d_slope_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals