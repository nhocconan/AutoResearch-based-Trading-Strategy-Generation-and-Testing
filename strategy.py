#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper channel with 1w EMA50 uptrend and volume spike
# Short when price breaks below 1d Donchian lower channel with 1w EMA50 downtrend and volume spike
# Exit when price crosses the 1d Donchian midline (midpoint of upper/lower) or reverses to opposite channel
# Uses Donchian channels for trend-following breakouts, EMA for trend filter, volume for conviction
# Designed to capture major trends in both bull and bear markets with controlled frequency
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_Donchian_20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper channel, EMA50 uptrend, volume spike
            if (close[i] > upper[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, EMA50 downtrend, volume spike
            elif (close[i] < lower[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses midline or reverses to lower channel
            if (close[i] <= middle[i]) or (close[i] < lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses midline or reverses to upper channel
            if (close[i] >= middle[i]) or (close[i] > upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals