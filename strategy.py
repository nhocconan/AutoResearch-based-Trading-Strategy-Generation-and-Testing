#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with weekly EMA trend and volume confirmation
# Long when price breaks above Donchian upper band with weekly EMA uptrend and volume > 1.5x average
# Short when price breaks below Donchian lower band with weekly EMA downtrend and volume > 1.5x average
# Exit when price reverses to the Donchian middle band or opposite band
# Weekly EMA trend filter reduces whipsaw, volume confirmation ensures conviction
# Target: 30-80 total trades over 4 years (7-20/year) with size 0.25-0.30

name = "1d_Donchian_Breakout_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, weekly EMA uptrend, volume confirmation
            if (close[i] > upper_band[i] and 
                ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, weekly EMA downtrend, volume confirmation
            elif (close[i] < lower_band[i] and 
                  ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reverses to middle band or breaks below lower band
            if (close[i] <= middle_band[i]) or (close[i] < lower_band[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverses to middle band or breaks above upper band
            if (close[i] >= middle_band[i]) or (close[i] > upper_band[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals